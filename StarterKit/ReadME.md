The purpose of this starter kit is to provide a script that can be generalizable for most machine learning tasks. 
This should not be mistaken for something that can directly be used on any dataset. Machine learning isn't that easy. 
I made this script using many popular python packages for purposes that come after the data exploration,
feature engineering, and understanding of the dataset is done by the researcher (probably in R or any other statistical software).
This would marely help you in speeding up your process of by not thinking much about starting off with a blank 
python script. By most, I would like to specify the following assumptions that need to be satisfied before using this 
on any dataset.

Assumptions
===========
- All variables must be numerical in 'nature': Yes. Since I'm using scikit-learn package in my code, this is a 
necessity. This means that all categorical/textual variables must be cleaned and coded appropriately before 
implementation.

- The dataset to be used must just have the target variable and features: Any other variables like ID, date(if not a 
feature) should be removed. You can always hardcode this stuff too.

- Should contain just one target variable column. 

P.S. This is a work in progress.



