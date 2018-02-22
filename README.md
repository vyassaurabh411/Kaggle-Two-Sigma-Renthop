# Kaggle-Two-Sigma-Renthop
Repository contains code that I created for Two Sigma Connect: Rental Listing Inquiries Kaggle competition. 
Two Sigma hosted a recruiting competition featuring rental listing data from RentHop. Kagglers were asked to predict the number of inquiries a new listing receives based on the listing’s creation date and other features. Doing so would help RentHop better handle fraud control, identify potential listing quality issues, and allow owners and agents to better understand renters’ needs and preferences.


I was ranked in top 2% in this competition (Leaderboard : https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/leaderboard). The dataset had 110,000 train + ~180,000 test cases with structured data, free text with HTML tags to clean up, images, geo-coordinates, and time information across the year. 
I created a 3 layered stacking architecture in with robust cross-validation scheme. The final solution included XGBoost as the most used model in the first layer and was supported with various other diverse models such as logistic regression, random forest, and neural networks.  
