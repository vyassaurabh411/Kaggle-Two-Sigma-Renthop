seed = 411

#Load the libraries
require(tidyverse)
require(tidytext)
require(stringr)
require(devtools)
require(caret)
require(parallel)
require(doSNOW)
require(Metrics)
require(xgboost)
require(rjson)
require(lubridate)
require(ggmap)
require(RecordLinkage)


## reading the data ---------------------------
train <- fromJSON(file = "train.json")
test <- fromJSON(file = "test.json")


## data processing and feature engineering ---------------------------

# unlist every variable except `photos` and `features` and convert to tibble
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)
glimpse(train)

train$interest_level <- factor(train$interest_level, 
                               levels = c("low", "medium", "high"))
y <- as.integer(train$interest_level)-1
train$interest_level <- NULL
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)


n_train <- nrow(train)
train_test=rbind(train,test)

# cleaning ba and br 
train_test$bathrooms[train_test$bathrooms == 10 | train_test$bathrooms == 20] <- train_test$bathrooms/10
train_test$bathrooms[train_test$bathrooms == 112] <- 2

# obtain missing lat lon - modularize this code
lat_lon_missing <- train_test$street_address[train_test$latitude == 0 | train_test$longitude == 0]

for (i in 1:length(lat_lon_missing)){
  temp <- geocode(paste(lat_lon_missing[i], "new york"), source = "google")
  train_test$longitude[train_test$street_address == lat_lon_missing[i]] <- temp$lon[1]
  train_test$latitude[train_test$street_address == lat_lon_missing[i]]  <- temp$lat[1]
}

# distance from city center, upper manhattan, empire state, times square and other important places
# this part of the code is to be changed as external data is not allowed in the competition
places <- c("Downtown Manhattan","Midtown Manhattan", "Upper Manhattan", 
            "Northern Brooklyn", "South Brooklyn",
            "Southwestern Brooklyn", "Eastern Brooklyn", "Southeastern Brooklyn",
            "Northeastern Queens", "Southeastern Queens",
            "Southwestern Queens", "West Bronx", "East Bronx", "Jersey City", 
            "Staten Island", "Brooklyn", "Queens","Bronx", "City Center"
)

ny_points <- data.frame(lon = 0, lat = 0)
for (i in 1:length(places)){
  ny_points <- geocode(paste(places[i], "New York"), source = "google")
  train_test[,ncol(train_test)+1] <- 
    mapply(function(lon, lat) (abs(lon - ny_points[[1]])  + abs(lat - ny_points[[2]])),
           train_test$longitude,
           train_test$latitude) 
  names(train_test)[ncol(train_test)] <- paste0("distance_", places[i])   
}

train_test$min_dist <- apply(train_test[,grepl("distance_",names(train_test))],1,min)
train_test$price_min_dist <- train_test$price*train_test$min_dist
area_name <- colnames(train_test[,grepl("distance_",
                                        names(train_test))])[apply(train_test[,grepl("distance_",names(train_test))],1,which.min)]
train_test$area_name <- substr(area_name, 10, nchar(area_name))


price_by_area <- train_test %>%
  group_by(area_name, bedrooms, bathrooms) %>%
  summarise(medianprice = median(price), meanprice = mean(price))

train_test <- train_test %>%
  left_join(price_by_area) %>%
  mutate(price_diff_mean = price - meanprice, price_ratio_mean = price/meanprice,
         price_diff_median = price - medianprice, price_ratio_median = price/medianprice)


rm(price_by_area, area_name, ny_points, lat_lon_missing, places)

# convert created to date time and new features from date
train_test$created <- ymd_hms(train_test$created, tz = "America/New_York")

train_test <- train_test %>%
  mutate(day = day(created), month = month(created), 
         hour = hour(created),weekday = wday(created)) %>%
  mutate(pricebed = ifelse(bedrooms == 0,price,price/bedrooms),  
         pricebath = ifelse(bathrooms == 0,price,price/bathrooms)) 


# new features : sum of rooms, bathrooms and price per room per bathroom
train_test$room_sum <- train_test$bedrooms + train_test$bathrooms + 1
train_test$room_diff <- train_test$bedrooms - train_test$bathrooms
train_test$room_price <- train_test$price/train_test$room_sum
train_test$bed_ratio <- train_test$bedrooms/train_test$room_sum

# new features : feature count, length of photos, char in description
train_test$feature_count <- lengths(train_test$features)
train_test$photo_count <- lengths(train_test$photos)
train_test$desc_len <- nchar(train_test$description)

# new feature : distance between display address and street address
train_test$address_similarity <- levenshteinSim(tolower(train_test$street_address),
                                                tolower(train_test$display_address))


# convert character features to numericals
char_vars <- c("building_id", "manager_id","street_address",
               "display_address", "area_name")
train_test <- map_at(train_test, char_vars, as.factor) %>%  
  map_if(is.factor, as.integer) %>% 
  tibble::as_tibble(.)

# adding manager and building count and rank
manager_rank <- train_test %>%
  group_by(manager_id) %>%
  summarise(manager_count = n()) %>%
  mutate(manager_rank = min_rank(desc(manager_count)))%>%
  arrange(manager_rank)


building_rank <- train_test %>%
  group_by(building_id) %>%
  summarise(building_count = n()) %>%
  mutate(building_rank = min_rank(desc(building_count)))%>%
  arrange(building_rank)

train_test <- train_test %>%
  left_join(manager_rank) %>%
  left_join(building_rank)


cut_off = 35
mgr_casted <- train_test %>%
  mutate(manager = paste0("manager_", manager_id), manager = 
           ifelse(manager_count < cut_off, "manager_other", manager)) %>%
  count(listing_id, manager) %>%
  spread(manager, n, fill = 0)

building_casted <- train_test %>%
  mutate(building = paste0("building_", building_id), building = 
           ifelse(building_count < cut_off, "buidling_other", building)) %>%
  count(listing_id, building) %>%
  spread(building, n, fill = 0)


train_test <- train_test %>%
  left_join(mgr_casted) %>%
  left_join(building_casted)


# adding text features  
features_text <- train_test %>%
  select(listing_id, features) %>% 
  filter(map(features, is_empty) != TRUE) %>%
  unnest(features) %>%
  mutate(features = tolower(features))

index <- str_detect(features_text$features,('laundry|dryer|washer'))& 
  !str_detect(features_text$features,('dishwasher'))
features_text$features[index] <- "Laundry"

index <- str_detect(features_text$features,('roof'))
features_text$features[index] <- "Roof_deck"

index <- str_detect(features_text$features,('outdoor'))
features_text$features[index] <- "Outdoor_Space"

index <- str_detect(features_text$features,('war'))
features_text$features[index] <- "Pre-war"

index <- str_detect(features_text$features,('wood'))
features_text$features[index] <- "Hardwood"

index <- str_detect(features_text$features,('garden'))
features_text$features[index] <- "Garden"

index <- str_detect(features_text$features,('pool'))
features_text$features[index] <- "Swimming_Pool"

index <- str_detect(features_text$features,('fitness|gym'))
features_text$features[index] <- "Fitness"

index <- str_detect(features_text$features,('high ceilings|high ceiling'))
features_text$features[index] <- "High_Ceilings"

index <- str_detect(features_text$features,('cat|dog|pet')) &
  !str_detect(features_text$features,('no pets'))
features_text$features[index] <- "Pets_Allowed"

index <- str_detect(features_text$features,('doorman'))
features_text$features[index] <- "Doorman"

index <- str_detect(features_text$features,('terrace'))
features_text$features[index] <- "Terrace"

index <- str_detect(features_text$features,('parking'))
features_text$features[index] <- "Parking"


features_top_words <- features_text %>%
  count(features, sort = T) %>%
  top_n(45)
# print(features_top_words, n=45)


feature_dtm <- features_text %>%
  filter((features %in% features_top_words$features)) %>%
  mutate(features = as.factor(features)) %>%
  count(listing_id, features)
feature_dtm <- spread(feature_dtm, features, n, fill = 0)


train_test <- train_test %>%
  left_join(feature_dtm, by = "listing_id") 

train_test[is.na(train_test)] <- 0


# remove unwanted variables
train_test_1 <- train_test %>%
  select(-features,-photos, -description, -created)

# break into train test again and write the files
train <- train_test_1[1:n_train,]
test <- train_test_1[(n_train + 1): nrow(train_test),]

label <- train$y
train$y <- NULL

## creating xgb model ---------------------------

params=list(objective = 'multi:softprob',
            eta=0.005, 
            max_depth= 6,
            subsample=.9,
            colsample_bytree=.35,
            min_child_weight=1,
            gamma = 2,
            num_class = 3,
            eval_metric= "mlogloss"
)

set.seed(seed)
nfold <- 5
folds = createFolds(label, k = nfold, returnTrain = T, list = T)
n_test <- nrow(test)
prediction = data.frame(low = rep(0,n_test), medium = rep(0,n_test), 
                        high = rep(0,n_test))

OOB_pred = data.frame(listing_id = NULL, high = NULL, 
                      medium = NULL, low = NULL)

for (i in 1:length(folds)){
  train_x <- train[folds[[i]],]
  train_y <- label[folds[[i]]]
  val_x <- train[-folds[[i]],]
  val_y <- label[-folds[[i]]]
  id_val <- train$listing_id[-folds[[i]]]
  
  dtrain <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)
  dVal <- xgb.DMatrix(data = as.matrix(val_x), label = val_y)
  dtest <- xgb.DMatrix(data = as.matrix(test))
  cat("\n\nStarting fold: ", i, "\n\n")
  model = xgb.train(params = params, data = dtrain, nrounds = 50000, 
                    early_stopping_rounds=50,
                    print_every_n = 1000,
                    verbose= 1, seed = seed, 
                    watchlist = list(train=dtrain, test = dVal),maximize=FALSE)
  
  pred = as.data.frame(matrix(predict(model,dtest), nrow = nrow(test), byrow = T))
  prediction = prediction + pred
  Val_pred=as.data.frame(matrix(predict(model,dVal), nrow = length(val_y), byrow = T))
  names(Val_pred) <- c("low", "medium", "high")
  # val_logloss = mLogLoss(cbind(listing_id=id_val,Val_pred),val_y)
  # cat("\n\nlog loss in fold", i,":",val_logloss,"\n\n")
  OOB_pred =rbind(OOB_pred,cbind(listing_id=id_val,Val_pred))
}

prediction = data.frame(listing_id = as.integer(test$listing_id), 
                        prediction/nfold)
prediction <- prediction[,c("listing_id", "high","medium","low")]
dim(prediction)
head(prediction)

OOB_pred <- OOB_pred[,c("listing_id", "high","medium","low")]
OOB_pred$listing_id <- as.integer(OOB_pred$listing_id)
dim(OOB_pred)
head(OOB_pred)

write_csv(prediction,paste0(Sys.Date(),"-XGB_1_Test_5Fold_CV_0.5284.csv"))
write_csv(OOB_pred,paste0(Sys.Date(),"-XGB_1_OOB_5Fold_CV_0.5284.csv"))

## Generate Feature Importance Plot
imp <- xgb.importance(names(train),model = model)
imp[1:50,]
xgb.ggplot.importance(imp)

