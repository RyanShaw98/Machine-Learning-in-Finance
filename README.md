# *AWS Server has been taken down*

# Machine Learning in Finance
The project provides new investors with an application that provides necessary information to help them make smart investment decisions without becoming overwhelmed by an abundance of advanced tools and features. 
It displays numerous aspects of stock data using interactive charts and graphs; the main stock aspect being a forecast of a user selected stock's close prices two weeks into the future. 
Other aspects are the traditional open, high, low, close and volume values, as well as a fifty day moving average to allow the user to see past trends.
Finally, it provides several quality of life improvements, such as congregating the latest and hottest news relevant to stock and crypto, as well as a built in notepad where the user can import and export their notes without leaving the application.

## Table of Contents
* [Getting Started](#getting-started)
* [User Guide](#user-guide)
* [Machine Learning](#machine-learning)
* [Further Documentation](#further-documentation)
* [License](#license)
* [Authors](#authors)
* [References](#references)

## Getting Started
Start by installing the [prerequisites](#prerequisites) and the [required modules](#install-libraries-and-run-program), this can be done using [pip](https://docs.python.org/3/installing/index.html) via the 
command prompt. Once the prerequisites and modules are installed, the project can be downloaded from [Gitlab](https://cseegit.essex.ac.uk/ce301_2019/ce301_shaw_r). It can then be ran via command prompt 
or your preferred IDE.

### Prerequisites
The prerequisites are:
* Python 3

### Installing
The project itself does not need to be installed, the .py script can be downloaded and ran. However, you must install the required modules detailed [below](#install-libraries-and-run-program). You can then 
either run the program through the command prompt, or by compiling it in your preferred IDE (such as IDLE, PyCharm etc.).

#### Install libraries and run program
Below are the necessary Python modules and versions (different versions may be compatible, but I have included my versions for certainty) required to run the program, and finally the command to run the 
program itself independent of an IDE. If you wish, you can open and run the project in an IDE instead.

```
pip install matplotlib==3.1.1
pip install numpy==1.17.2
pip install pandas==0.25.1
pip install pandas-datareader==0.8.1
pip install PyQt5==5.13.0
pip install mpl-finance==0.10.0
pip install newsapi-python==0.2.5
pip install scikit-learn==0.21.3
pip install mysql-conector==2.2.9
pip install bcrypt==3.1.7
pip install mysql-connector-python==2.2.9
pip install crypto-news-api==2.3.1
pip install Keras==2.3.1
pip install tensorflow==2.1.0
python ML_in_Finance.py
```

### Running Tests
To run tests based on the GUI, enter the line below into the command prompt:

```
python test_ML_in_Finance.py
```
Alternatively, you can run "test_ML_in_Finance.py" in an IDE.

Testing of the machine learning components can be found in the Colab notebooks on Gitlab.

## GUI

### Login

This is the first GUI you will be met with, simply enter your credentials or register for an account. Login credentials are validated against the database.

![](https://i.gyazo.com/9ec73ee0b3ec81a1c907d8e9c7c93ead.png)

### Register

This is the window that will open when the user registers for a new account. Numerous validation techniques are in place to assure a valid email is entered, passwords match and that the user has not already registered under the entered email.

![](https://i.gyazo.com/a9aee569cf6a5f62e028dc5454fd5891.png)

### Portfolio

The portfolio allows you to view and add assets to your portfolio. It shows several statistics such as the net and gross profit, as well as your portfolio distribution in a pie chart. Portfolios are saved in the database so they will be available regardless of the PC used.

![](https://i.gyazo.com/568af0652c9a367520afe01742a654f7.png)

### Stocks

#### Comparison Graph
The comparison graph lets you plot the stock data of different companies against each other. The data that can be plotted consists of open, high, low, close (these can also be plotted in a candlestick format), volume, moving average and forecast. For more information on how the forecast is calculated, click [here](#machine-learning) 

![](https://i.gyazo.com/cf502d985dc0b9a79f2e2a366d5bbd67.png)

To add a stock to the comparison graph, first look in the "Add a Company" box and select the start and end dates for the stock data.

Next, put in the ticker symbol for the company you want to compare and clicked the "Add" button; the data for your chosen company will now be plotted.

If the graph is too cluttered, go to the "Plotted Data" box. Here you can select what data is visible, and what is hidden. In the above figure, you can see the "Close", "Moving Average" and (if you were to scroll down) "Forecast" are visible.

To remove a company from the graph, simply double click their ticker symbol in the "Plotted Tickers" box.

To zoom in/out and move the graph around, use the Matplotlib tool bar on the bottom left of the application.

#### Stock Data Table
The stock data table plots a stock's ticker symbol, open, high, low, close and volume values.

![](https://i.gyazo.com/40a5b17ee6b18cd57cc6444a6171ec14.png)

Currently, only the FTSE 100 is listed.

All columns are sortable by clicking on a columns header - once for ascending, and twice for descending.

If you double click on a company's ticker symbol in the table, it opens a Yahoo Finance web page for the corresponding ticker.

#### News Feed
The news feed displays the latest headlines from the 'business' category from various news sources.

![](https://i.gyazo.com/6cf2704c1955decdf181211c60efbb3b.png)
![](https://i.gyazo.com/04fbac44839de670d3fb0720f527d8c0.png)

The headline for each article is listed in a row, and at the end of the row it tells you how long ago the article was published.

If you double click the headline, it opens up the whole article in a web page.

#### Notes
The notes tab lets you save and open simple text notes.

![](https://i.gyazo.com/47607ed496bd0e2f0eed99d92b745f68.png)

Clicking "Open" will use your operating systems natural file picker, where the text contents of the chosen file will be loaded into the application ready for editing.

Clicking "Save" will again use your operating systems natural directory picker to choose where and what to save your notes as.

### Crypto

The crypto section is very similar to the stock section in terms of the comparison graph, data table, news and notes; although, the data table is populated with cryptocurrencies and the news is related to crypto.

The main difference is the movement prediction. This uses deep learning to predict the movement magnitude and direction of Bitcoin, Ethereum, Ripple, Litecoin and Bitcoin Cash over the next ten days.

![](https://i.gyazo.com/169e2dbe0a9cddb92815a897e1230007.png)

### Your Account

The account page shows you what details are stored in regards to your account, and gives you the ability to modify any details, log out or delete the account altogether.

![](https://i.gyazo.com/a1da383526c913a3e8120fb8e5a34aa2.png)

## Artificial Intelligence

### Data

The dataset used is specific to each stock or cryptocurrency. Where possible the previous ten years of open, high, low, close and volume data is obtained for each user requested asset from [Yahoo Finance](https://uk.finance.yahoo.com/). For each dataset, the following features have been engineered:

* ```HL_Pct_diff```: Percentage difference between daily high and low prices
* ```OC_Pct_Diff```: Percentage difference between daily open and close prices
* ```Close```: The daily close price

### Machine Learning

Supervised learning consists of features and labels; features are inputs and labels are the answers to the inputs. Data is split into separate sets for training and testing, the testing set teaches the model what it should output for a given input, and the testing set evaluates how well the model has learnt. The goal is to sufficiently teach the model to a point where it can predict the outcome for inputs it has never seen.

As the aim of the project is forecasting stock prices, we are dealing with a regression problem. In regression problems the model is predicting a value (e.g. a stock's price), as opposed to a class or cluster.

#### High Level Flowchart
![](https://i.gyazo.com/a29c2472b2f9ac74a87229918b6771a7.png)

#### Comparing Models' Score
From the below figures, the Linear Regression model was found to be the best. On the other hand, Elastic Net clearly proved to be the worst.

![](https://i.gyazo.com/d827149b0fb25f58f8003803d57d28a7.png)
![](https://i.gyazo.com/319b100a70b1c152d83e24d1c4406c01.png)
![](https://i.gyazo.com/f362fa9f86064316b3488cb56de2a4a4.png)
![](https://i.gyazo.com/5928ed6730d57c4ea6b3a43c7e835e93.png)
![](https://i.gyazo.com/5e69ca5ad919ae3562608775e8c4a833.png)
![](https://i.gyazo.com/7cffe4ebe5623ec737a48e8aa5f3c9dc.png)

### Deep Learning

A neural network is unlike a machine learning model, in the way that it is not an algorithm, but a framework inspired by biological neural networks that enables a series of algorithms to work together.

A dense layer is one of the more simplistic layers; it's fully connected, meaning all neurons in the previous layer are connected to all nodes in the next layer, and its output is the product of the inputs and the weights (plus a bias if provided).

Long short-term memory is an artificial recurrent neural network (RNN), these handle time-series data very well which is why they are being used in this application. They have "LSTM cell blocks in place of standard neural network layers. These cells have various components called the input gate, the forget gate and the output gate" [[1](#references)].

![](https://i1.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/09/LSTM-diagram.png?w=669&ssl=1)

The data being fed into the neural network differs from that going into the machine learning models. The same data is obtained however, only the raw close prices are used; the close prices are split into thirty day sequences for the features, and the consecutive ten days as the labels.

## Database

Amazon Web Services' are used to host the cloud database that users' account details and portfolios are stored on. Sensitive user information is protected using bcrypt.

## Further Documentation
For more documentation including initial project planning and objectives see [here](https://cseegit.essex.ac.uk/ce301_2019/ce301_shaw_r/tree/master/documentation)

## Versioning Strategy
The project uses [semantic versioning](https://semver.org/), with the MVP being release 1.0.0

## License
This project is licensed under the GNU General Public License of 2007.

For more information please read the [license](https://cseegit.essex.ac.uk/ce301_2019/ce301_shaw_r/-/blob/master/LICENSE.txt) documentation.

## Authors
* Ryan Shaw

## References
* [1][LSTM Definition and diagram](https://adventuresinmachinelearning.com/keras-lstm-tutorial/)
* [Matplotlib](https://matplotlib.org/)
* [Mpl-Finance](https://matplotlib.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Pandas Data Reader](https://pandas.pydata.org/)
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro)
* [NewsApi-Python](https://newsapi.org/docs/client-libraries/python)
* [Scikit-Learn](https://scikit-learn.org/stable/)
