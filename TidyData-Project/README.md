# Tidy Data Project

<h2> In this project, I cleaned a raw dataset for better readability, perform exploratory data analysis, and create meaningful visualizations. </h2>
<p>
<h3> Goals: </h3>
My goals with this project were to practice tidying data, as this is a large component of data analysis as a whole. This primarily involves following 
  the following specific structure: each variable is a column, each observation is a row, and each type of observational unit is a table. This format 
  allows for datasets that are easy to manipulate and model.
<p>
<h3> Process: </h3>
The original dataset did not follow the tidy data structure, so I first had to use the pd.melt() function from pandas to convert the dataset from a 
  wide to a long format. After doing this, the columns did not follow all of the tidy data principles, so I had to complete the conversion of the table
  into a format that follows the tidy data principles. I then created visualizations of the tidy data, because data alone cannot always adequately 
  reflect the relationships between variables. Finally, I also created an aggregate function that allowed for a concise, more general table of the 
  variables and data.
<p>
<h3> Dataset: </h3>
The dataset itself, it contained US federal research and development (R&D) funding for 14 departments from 1976 to 2017 and GDP values for each 
  corresponding year. 
<p>
<h3> Visualization: </h3>
