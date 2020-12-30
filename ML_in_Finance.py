import datetime as dt
import pickle
import sys
import webbrowser

import bcrypt
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import mysql.connector
import numpy
import pandas
import pandas_datareader.data as pdr
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from crypto_news_api import CryptoControlAPI
from matplotlib import style
from matplotlib import ticker as mpl_ticker
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_finance import candlestick_ohlc
from newsapi import NewsApiClient
from pandas.plotting import register_matplotlib_converters
from pandas_datareader._utils import RemoteDataError
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow_core.python.keras.models import load_model


register_matplotlib_converters()
style.use('ggplot')
pandas.set_option('display.max_columns', None)  # Show all data frame columns
pandas.set_option('display.width', 150)  # Print data frame on a single line

STOCK_NEWS_API = NewsApiClient(api_key='dbbb195f3d35453b9e6cfd72f5d51d7b')
CRYPTO_NEWS_API = CryptoControlAPI("83911bda820a145e1421297d3b1e5abf")

DB_HOST_NAME = "ce301.ceyjwmghheko.us-east-2.rds.amazonaws.com"
DB_NAME = "financeportfolio"
DB_USERNAME = "rs16212"
DB_PASSWORD = "Tr7d_A6iQUW*ike$?08f"
DB = mysql.connector.connect(host=DB_HOST_NAME, user=DB_USERNAME, passwd=DB_PASSWORD, database=DB_NAME)
DB_CURSOR = DB.cursor()

USER_ID = -1
USER_EMAIL = -1
USER_FIRSTNAME = -1
USER_LASTNAME = -1
USER_PASSWORD_LENGTH = -1

FTSE_100_TICKERS = []
SP_500_TICKERS = []
CRYPTO_TICKERS = []


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("ui/portfolio.ui", self)
        self.show()
        self.setWindowIcon(QtGui.QIcon("ui/images/python-icon.png"))
        self.setWindowTitle("Machine Learning in Finance")

        set_ftse_100_tickers()
        set_sp_500_tickers()
        set_crypto_tickers()

        # Plot graphs

        self.plotted_lines = {}
        self.plotted_lines_crypto = {}

        # Portfolio chart

        self.pie_chart_labels = []
        self.pie_chart_sizes = []
        self.fig_portfolio, self.ax_portfolio = plt.subplots()
        self.fig_portfolio.patch.set_facecolor('xkcd:grey')
        self.ax_portfolio.pie(self.pie_chart_sizes, labels=self.pie_chart_labels, autopct='%1.1f%%', startangle=90)
        self.ax_portfolio.axis('equal')
        self.widgetPortfolioChart = FigureCanvas(self.fig_portfolio)
        lay_portfolio = QtWidgets.QVBoxLayout(self.assetChart)
        lay_portfolio.setContentsMargins(0, 0, 0, 0)
        lay_portfolio.addWidget(self.widgetPortfolioChart)

        # Stock Graphs

        # Plot comparison and volume graph
        self.fig, (not_needed, not_needed2) = plt.subplots(nrows=2, ncols=1)
        self.fig.patch.set_facecolor('xkcd:grey')
        # Allocate graph positions
        self.ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
        self.ax2 = plt.subplot2grid((10, 1), (7, 0), rowspan=3, colspan=1, sharex=self.ax1)
        self.ax1.xaxis_date()  # Displays axis label as normal date rather than mpl_date
        self.ax1.set_ylabel("Price", color='white')
        self.ax2.xaxis_date()
        self.ax2.get_yaxis().set_major_formatter(
            mpl_ticker.FuncFormatter(lambda x, pos: format(x, ',')))  # Format ax2 y-axis scale
        self.ax2.set_ylabel("Volume", color='white')
        self.ax1.grid(alpha=0.5)
        self.ax2.grid(alpha=0.5)
        self.ax1.tick_params(labelcolor='white')
        self.ax2.tick_params(labelcolor='white')
        self.fig.tight_layout()

        self.widgetComparison = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self.plotComparisonGraph)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.widgetComparison)
        self.toolbar = NavigationToolbar(self.widgetComparison, self)
        self.toolbar.setStyleSheet("background-color: gray;")
        self.addToolBar(QtCore.Qt.BottomToolBarArea, self.toolbar)
        self.toolbar.hide()

        # Crypto Graphs

        # Plot comparison and volume graph
        self.fig_crypto, (not_needed_crypto, not_needed2_crypto) = plt.subplots(nrows=2, ncols=1)
        self.fig_crypto.patch.set_facecolor('xkcd:grey')
        # Allocate graph positions
        self.ax1_crypto = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
        self.ax2_crypto = plt.subplot2grid((10, 1), (7, 0), rowspan=3, colspan=1, sharex=self.ax1_crypto)
        self.ax1_crypto.xaxis_date()  # Displays axis label as normal date rather than mpl_date
        self.ax1_crypto.set_ylabel("Price", color='white')
        self.ax2_crypto.xaxis_date()
        self.ax2_crypto.get_yaxis().set_major_formatter(
            mpl_ticker.FuncFormatter(lambda x, pos: format(x, ',')))  # Format ax2 y-axis scale
        self.ax2_crypto.set_ylabel("Volume", color='white')
        self.ax1_crypto.grid(alpha=0.5)
        self.ax2_crypto.grid(alpha=0.5)
        self.ax1_crypto.tick_params(labelcolor='white')
        self.ax2_crypto.tick_params(labelcolor='white')
        self.fig_crypto.tight_layout()

        self.widgetComparisonCrypto = FigureCanvas(self.fig_crypto)
        lay_crypto = QtWidgets.QVBoxLayout(self.plotComparisonGraphCrypto)
        lay_crypto.setContentsMargins(0, 0, 0, 0)
        lay_crypto.addWidget(self.widgetComparisonCrypto)
        self.toolbar_crypto = NavigationToolbar(self.widgetComparisonCrypto, self)
        self.toolbar_crypto.setStyleSheet("background-color: gray;")
        self.addToolBar(QtCore.Qt.BottomToolBarArea, self.toolbar_crypto)
        self.toolbar_crypto.hide()

        # Setting Default Component Values

        read_ftse100_pickle = open("stock_data/ftse100tickers.pickle", "rb")
        ftse100_symbols = pickle.load(read_ftse100_pickle)

        read_sp500_pickle = open("stock_data/sp500tickers.pickle", "rb")
        sp500_symbols = pickle.load(read_sp500_pickle)

        read_crypto_pickle = open("crypto_data/top_crypto_symbols.pickle", "rb")
        crypto_symbols = pickle.load(read_crypto_pickle)

        stock_completer = QtWidgets.QCompleter(ftse100_symbols + sp500_symbols)
        stock_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        crypto_completer = QtWidgets.QCompleter(crypto_symbols)
        crypto_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        stock_and_crypto_completer = QtWidgets.QCompleter(ftse100_symbols + sp500_symbols + crypto_symbols)
        stock_and_crypto_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)

        # Portfolio

        self.labelPortfolioTitle.setText(f"{USER_FIRSTNAME}'s Portfolio")
        self.initialise_portfolio_from_database()
        self.tableAssets.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.lineEditAssetSymbol.setCompleter(stock_and_crypto_completer)

        # Stocks

        self.tableComparisonTickers.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.dateEditStartDate.setDate(QtCore.QDate.currentDate().addMonths(-3))
        self.dateEditEndDate.setDate(QtCore.QDate.currentDate())
        self.lineEditCompany.setCompleter(stock_completer)

        # Crypto

        self.tableComparisonCrypto.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.dateEditStartDateCrypto.setDate(QtCore.QDate.currentDate().addMonths(-3))
        self.dateEditEndDateCrypto.setDate(QtCore.QDate.currentDate())
        self.lineEditCrypto.setCompleter(crypto_completer)
        self.set_crypto_dl_predictions()

        # Account

        self.labelActualName.setText(f"{USER_FIRSTNAME} {USER_LASTNAME}")
        self.labelActualEmail.setText(f"{USER_EMAIL}")
        self.labelActualPassword.setText("•" * USER_PASSWORD_LENGTH)

        # Populate tables

        # Stocks

        self.populate_table(self.tableTickerData, "stock")

        self.top_business_headlines = STOCK_NEWS_API.get_top_headlines(country='gb', category='business', language='en')
        self.top_business_articles = pandas.DataFrame(self.top_business_headlines['articles'])
        self.populate_news_table(self.tableNews, self.top_business_articles, "stock")

        # Crypto

        self.populate_table(self.tableCryptoData, "crypto")

        self.top_crypto_headlines = CRYPTO_NEWS_API.getTopNews()
        self.populate_news_table(self.tableNewsCrypto, self.top_crypto_headlines, "crypto")

        # Events

        self.toolButtonMenu.clicked.connect(self.change_menu_visibility)
        self.btnPortfolio.clicked.connect(self.switch_window_portfolio)
        self.btnStocks.clicked.connect(self.switch_window_stocks)
        self.btnCrypto.clicked.connect(self.switch_window_crypto)
        self.btnAccount.clicked.connect(self.switch_window_account)

        # Portfolio

        self.btnRefreshPortfolio.clicked.connect(self.refresh_portfolio_asset_values)
        self.btnAddAsset.clicked.connect(self.get_gui_input_and_add_to_portfolio)
        self.tableAssets.doubleClicked.connect(self.remove_portfolio_asset)

        # Stocks

        self.btnAddCompany.clicked.connect(self.add_comparison_ticker)
        self.lineEditCompany.returnPressed.connect(self.add_comparison_ticker)

        self.checkBoxOpen.stateChanged.connect(self.change_open_line_visibility)
        self.checkBoxHigh.stateChanged.connect(self.change_high_line_visibility)
        self.checkBoxLow.stateChanged.connect(self.change_low_line_visibility)
        self.checkBoxClose.stateChanged.connect(self.change_close_line_visibility)
        self.checkBoxMovingAverage.stateChanged.connect(self.change_ma_line_visibility)
        self.checkBoxForecast.stateChanged.connect(self.change_forecast_line_visibility)
        self.checkBoxCandlestick.stateChanged.connect(self.change_candlestick_line_visibility)

        self.tableComparisonTickers.doubleClicked.connect(self.remove_comparison_ticker)
        self.tableNews.doubleClicked.connect(self.open_stock_news_link)
        self.tableTickerData.doubleClicked.connect(self.open_yahoo_link_stock)
        self.btnSaveNotes.clicked.connect(self.save_notes)
        self.btnOpenNotes.clicked.connect(self.open_notes)

        # Crypto

        self.btnAddCrypto.clicked.connect(self.add_comparison_crypto)
        self.lineEditCrypto.returnPressed.connect(self.add_comparison_crypto)

        self.checkBoxOpenCrypto.stateChanged.connect(self.change_open_line_visibility_crypto)
        self.checkBoxHighCrypto.stateChanged.connect(self.change_high_line_visibility_crypto)
        self.checkBoxLowCrypto.stateChanged.connect(self.change_low_line_visibility_crypto)
        self.checkBoxCloseCrypto.stateChanged.connect(self.change_close_line_visibility_crypto)
        self.checkBoxMovingAverageCrypto.stateChanged.connect(self.change_ma_line_visibility_crypto)
        self.checkBoxForecastCrypto.stateChanged.connect(self.change_forecast_line_visibility_crypto)
        self.checkBoxCandlestickCrypto.stateChanged.connect(self.change_candlestick_line_visibility_crypto)

        self.tableComparisonCrypto.doubleClicked.connect(self.remove_comparison_crypto)
        self.tableNewsCrypto.doubleClicked.connect(self.open_crypto_news_link)
        self.tableCryptoData.doubleClicked.connect(self.open_yahoo_link_crypto)

        # Account

        self.btnUpdateName.clicked.connect(self.update_users_name)
        self.btnUpdateEmail.clicked.connect(self.update_users_email)
        self.btnUpdatePassword.clicked.connect(self.update_users_password)
        self.btnDeleteAccount.clicked.connect(self.delete_account)
        self.btnLogout.clicked.connect(self.log_out)

    def set_crypto_dl_predictions(self):
        currencies = ["BTC-GBP", "ETH-GBP", "XRP-GBP", "LTC-GBP", "BCH-GBP"]
        gui_labels = [self.labelBTCMovement, self.labelETHMovement, self.labelXRPMovement, self.labelLTCMovement, self.labelBCHMovement]
        pixmap_labels = [self.labelBTCMovementDirection, self.labelETHMovementDirection, self.labelXRPMovementDirection, self.labelLTCMovementDirection, self.labelBCHMovementDirection]
        for currency, label, pixmap in zip(currencies, gui_labels, pixmap_labels):
            start = dt.datetime(2010, 1, 27)
            end_date = dt.datetime(2020, 1, 27)

            df = pdr.get_data_yahoo(currency, start, end_date)
            df = df[["Close"]]
            scaler = MinMaxScaler()

            df = pandas.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

            # How many days looking back to train
            days_to_train = 30

            # Features (only one feature: price)
            num_of_features = 1

            # Obtaining the last sequence
            X = numpy.array(list(df.Close)[-40:-10])

            X = X.reshape(X.shape[0], num_of_features)

            model = load_model(f"deep_learning_models/{currency}_model.h5")

            # Getting predictions by predicting from the last available X variable
            scaled_predictions = model.predict(X.reshape(1, days_to_train, num_of_features)).tolist()[0]

            # Transforming values back to their normal prices
            unscaled_predictions = scaler.inverse_transform(numpy.array(scaled_predictions).reshape(-1, 1)).tolist()

            first_prediction = unscaled_predictions[0][0]
            last_prediction = unscaled_predictions[-1][0]

            pct_change = (abs(first_prediction - last_prediction) / first_prediction) * 100
            if first_prediction > last_prediction:
                pct_change = -pct_change

            label.setText(str(round(pct_change, 2)) + '%')
            if pct_change > 0:
                label.setStyleSheet("QLabel {border: none;color: green}")
                pixmap.setPixmap(QtGui.QPixmap('ui/images/green_up_arrow.png'))
            else:
                label.setStyleSheet("QLabel {border: none;color: red}")
                pixmap.setPixmap(QtGui.QPixmap('ui/images/red_down_arrow.png'))

    def update_users_name(self):
        new_firstname = self.lineEditUpdatedFirstname.text()
        self.lineEditUpdatedFirstname.setText("")
        new_lastname = self.lineEditUpdatedLastname.text()
        self.lineEditUpdatedLastname.setText("")

        global USER_FIRSTNAME, USER_LASTNAME
        new_firstname = new_firstname if len(new_firstname) > 0 else USER_FIRSTNAME
        new_lastname = new_lastname if len(new_lastname) > 0 else USER_LASTNAME
        USER_FIRSTNAME = new_firstname
        USER_LASTNAME = new_lastname

        insertion_values = (new_firstname, new_lastname, USER_ID,)
        DB_CURSOR.execute("UPDATE accounts SET firstname = %s, surname = %s WHERE id = %s", insertion_values)
        DB.commit()

        self.labelActualName.setText(f"{USER_FIRSTNAME} {USER_LASTNAME}")
        self.labelPortfolioTitle.setText(f"{USER_FIRSTNAME}'s Portfolio")
        QtWidgets.QMessageBox.information(self, "Updated", "Name updated successfully")

    def update_users_email(self):
        new_email = self.lineEditUpdatedEmail.text()
        self.lineEditUpdatedEmail.setText("")

        if len(new_email) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Email field is empty")
            return

        val = (new_email,)
        DB_CURSOR.execute("SELECT email FROM accounts WHERE email=%s", val)
        result = DB_CURSOR.fetchall()

        if new_email.find("@") != -1 and len(result) == 0:
            global USER_EMAIL
            new_email = new_email if len(new_email) > 0 else USER_EMAIL
            USER_EMAIL = new_email

            insertion_values = (USER_EMAIL, USER_ID,)
            DB_CURSOR.execute("UPDATE accounts SET email = %s WHERE id = %s", insertion_values)
            DB.commit()

            self.labelActualEmail.setText(f"{USER_EMAIL}")
            QtWidgets.QMessageBox.information(self, "Updated", "Email updated successfully")
        elif len(result) != 0:
            QtWidgets.QMessageBox.warning(self, "Error", f"An account already exists under {new_email}")
        else:
            QtWidgets.QMessageBox.warning(self, "Error", f"{new_email} is an invalid email address")

    def update_users_password(self):
        new_password = self.lineEditUpdatedPassword.text()
        self.lineEditUpdatedPassword.setText("")
        new_password_repeat = self.lineEditUpdatedPasswordRepeat.text()
        self.lineEditUpdatedPasswordRepeat.setText("")

        if len(new_password) == 0 or len(new_password_repeat) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Password field is empty")
            return

        if new_password == new_password_repeat:
            global USER_ID, USER_PASSWORD_LENGTH
            hashed_passwd = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
            val = (hashed_passwd, USER_ID,)
            DB_CURSOR.execute("UPDATE accounts SET password = %s WHERE id = %s", val)
            DB.commit()

            USER_PASSWORD_LENGTH = len(new_password)
            self.labelActualPassword.setText("•" * USER_PASSWORD_LENGTH)
            QtWidgets.QMessageBox.information(self, "Updated", "Password updated successfully")
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Passwords do not match")

    def log_out(self, confirmed):
        if not confirmed:
            log_out_account = QtWidgets.QMessageBox.question(self, "Log Out",
                                                             "Are you sure you want to log out of your account?")
            if log_out_account == QtWidgets.QMessageBox.Yes:
                self.close()
                self.login = LogInWindow()
        else:
            self.close()
            self.login = LogInWindow()

    def delete_account(self):
        del_acc = QtWidgets.QMessageBox.question(self, "Delete Account",
                                                 "Are you sure you want to delete your account?")
        if del_acc == QtWidgets.QMessageBox.Yes:
            global USER_ID
            insertion_values = (USER_ID,)
            DB_CURSOR.execute("DELETE FROM portfolios WHERE id=%s", insertion_values)
            DB.commit()
            DB_CURSOR.execute("DELETE FROM accounts WHERE id=%s", insertion_values)
            DB.commit()
            self.log_out(confirmed=True)

    def initialise_portfolio_from_database(self):
        global USER_ID
        DB_CURSOR.execute(f"SELECT asset, holding, paid FROM portfolios WHERE id={USER_ID}")
        for asset_name, amount_held, amount_paid in DB_CURSOR:
            self.add_asset_to_portfolio(asset_name, amount_held, amount_paid)

    def update_pie_chart(self):
        self.ax_portfolio.clear()
        self.ax_portfolio.pie(self.pie_chart_sizes, labels=self.pie_chart_labels, autopct='%1.1f%%', startangle=90)
        self.fig_portfolio.canvas.draw()

    def get_gui_input_and_add_to_portfolio(self):
        asset_symbol = self.lineEditAssetSymbol.text().upper()
        amount_held = self.lineEditAmountHeld.text()
        amount_paid = self.lineEditAmountPaid.text()

        try:
            amount_held = float(amount_held)
        except ValueError:
            QtWidgets.QMessageBox.about(self, "Error", f"'{amount_held}' is not a valid amount held")
            return
        try:
            amount_paid = float(amount_paid)
        except ValueError:
            QtWidgets.QMessageBox.about(self, "Error", f"'{amount_paid}' is not a valid amount paid")
            return

        valid_symbol = self.add_asset_to_portfolio(asset_symbol, amount_held, amount_paid)

        if valid_symbol:
            global USER_ID
            insertion_values = (USER_ID, asset_symbol, amount_held, amount_paid,)
            DB_CURSOR.execute("INSERT INTO portfolios (id, asset, holding, paid) VALUES (%s, %s, %s, %s)",
                              insertion_values)
            DB.commit()

            self.lineEditAmountHeld.setText("")
            self.lineEditAmountPaid.setText("")
            self.lineEditAssetSymbol.setText("")

    def add_asset_to_portfolio(self, asset_symbol, amount_held, amount_paid):
        date_today = dt.datetime.now().date()
        df_value_today = self.get_asset_data_from_yahoo(asset_symbol, date_today, date_today)
        if df_value_today is None:
            return False
        value_today = round(df_value_today["Close"].values[0] * amount_held, 2)
        profit_loss = round(value_today - amount_paid, 2)
        profit_loss_percentage = round(((value_today - amount_paid) / amount_paid) * 100, 2)

        r, g, b = (0, 220, 0) if profit_loss > 0 else (180, 0, 0)

        profit_loss_table_item = QNumericalTableWidgetItem(str(profit_loss))
        profit_loss_table_item.setForeground(QtGui.QBrush(QtGui.QColor(r, g, b)))

        profit_loss_percentage_table_item = QNumericalTableWidgetItem(str(profit_loss_percentage))
        profit_loss_percentage_table_item.setForeground(QtGui.QBrush(QtGui.QColor(r, g, b)))

        self.tableAssets.setSortingEnabled(False)
        self.tableAssets.insertRow(0)
        self.tableAssets.setItem(0, 0, QtWidgets.QTableWidgetItem(asset_symbol))  # Asset name
        self.tableAssets.setItem(0, 1, QNumericalTableWidgetItem(str(amount_held)))  # Amount held
        self.tableAssets.setItem(0, 2, QNumericalTableWidgetItem(str(amount_paid)))  # Amount paid
        self.tableAssets.setItem(0, 3, QNumericalTableWidgetItem(str(value_today)))  # Value
        self.tableAssets.setItem(0, 4, profit_loss_table_item)  # P/L
        self.tableAssets.setItem(0, 5, profit_loss_percentage_table_item)  # P/L %

        for column_count in range(self.tableAssets.columnCount()):
            self.tableAssets.item(0, column_count).setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableAssets.setSortingEnabled(True)

        self.pie_chart_labels.append(asset_symbol)
        self.pie_chart_sizes.append(value_today)
        self.update_pie_chart()

        self.update_portfolio_value_labels()

        return True

    def remove_portfolio_asset(self, item):
        table_item = self.tableAssets.findItems(item.data(), QtCore.Qt.MatchContains)
        table_item_row = table_item[0].row()
        removed_asset_symbol = self.tableAssets.item(table_item_row, 0).text()
        removed_amount_held = self.tableAssets.item(table_item_row, 1).text()
        removed_amount_paid = self.tableAssets.item(table_item_row, 2).text()
        self.tableAssets.removeRow(table_item_row)

        index_to_remove = self.pie_chart_labels.index(removed_asset_symbol)
        self.pie_chart_labels.pop(index_to_remove)
        self.pie_chart_sizes.pop(index_to_remove)
        self.update_pie_chart()

        self.update_portfolio_value_labels()

        global USER_ID
        insertion_values = (USER_ID, removed_asset_symbol, float(removed_amount_held), float(removed_amount_paid),)
        DB_CURSOR.execute("DELETE FROM portfolios WHERE id=%s AND asset=%s AND holding=%s AND paid=%s",
                          insertion_values)
        DB.commit()

    def update_portfolio_value_labels(self):
        total_value = 0
        total_paid = 0
        for row_num in range(self.tableAssets.rowCount()):
            total_value += float(self.tableAssets.item(row_num, 3).text())
            total_paid += float(self.tableAssets.item(row_num, 2).text())
        self.labelPortfolioGrossValue.setText(f"Portfolio Gross Value: £{round(total_value, 2)}")
        self.labelPortfolioNetValue.setText(f"Portfolio Net Profit: £{round(total_value - total_paid, 2)}")

    def refresh_portfolio_asset_values(self):
        self.pie_chart_labels = []
        self.pie_chart_sizes = []
        for row_num in range(self.tableAssets.rowCount()):
            asset_symbol = self.tableAssets.item(row_num, 0).text()
            amount_held = float(self.tableAssets.item(row_num, 1).text())
            amount_paid = float(self.tableAssets.item(row_num, 2).text())

            date_today = dt.datetime.now().date()
            df_value_today = self.get_asset_data_from_yahoo(asset_symbol, date_today, date_today)
            value_today = round(df_value_today["Close"].values[0] * amount_held, 2)
            profit_loss = round(value_today - amount_paid, 2)
            profit_loss_percentage = round(((value_today - amount_paid) / amount_paid) * 100, 2)

            self.tableAssets.item(row_num, 3).setText(str(value_today))
            self.tableAssets.item(row_num, 4).setText(str(profit_loss))
            self.tableAssets.item(row_num, 5).setText(str(profit_loss_percentage))

            self.pie_chart_labels.append(asset_symbol)
            self.pie_chart_sizes.append(value_today)
        self.update_pie_chart()

    def switch_window_portfolio(self):
        self.toolbar.hide()
        self.toolbar_crypto.hide()
        self.stackedWidget.setCurrentIndex(0)

    def switch_window_stocks(self):
        self.toolbar_crypto.hide()
        self.toolbar.show()
        self.stackedWidget.setCurrentIndex(1)

    def switch_window_crypto(self):
        self.toolbar.hide()
        self.toolbar_crypto.show()
        self.stackedWidget.setCurrentIndex(2)

    def switch_window_account(self):
        self.toolbar.hide()
        self.toolbar_crypto.hide()
        self.stackedWidget.setCurrentIndex(3)

    def change_menu_visibility(self):
        if self.frameMenu.isVisible():
            self.frameMenu.hide()
            self.toolButtonMenu.setArrowType(QtCore.Qt.ArrowType(4))
        else:
            self.frameMenu.show()
            self.toolButtonMenu.setArrowType(QtCore.Qt.ArrowType(3))

    def open_yahoo_link_stock(self, item):
        url = f"https://uk.finance.yahoo.com/quote/{self.tableTickerData.item(item.row(), 0).text()}"
        webbrowser.open_new_tab(url)

    def open_yahoo_link_crypto(self, item):
        url = f"https://uk.finance.yahoo.com/quote/{self.tableCryptoData.item(item.row(), 0).text()}"
        webbrowser.open_new_tab(url)

    def open_stock_news_link(self, item):
        row = self.top_business_articles.loc[self.top_business_articles['title'] == self.tableNews.item(item.row(), 0).text()]
        url = row['url'].tolist()[0]
        webbrowser.open_new_tab(url)

    def open_crypto_news_link(self, item):
        clicked_article_url = [article['url'] for article in self.top_crypto_headlines if
                               f"{article['title']} - {article['source']['name']}" == self.tableNewsCrypto.item(item.row(), 0).text()]
        webbrowser.open_new_tab(*clicked_article_url)

    def populate_news_table(self, table, articles, asset_type):
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        row_count = 0
        if asset_type == "stock":
            for index, row in articles.iterrows():
                table.insertRow(row_count)
                published = dt.datetime.strptime(row['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(row['title']))
                table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(self.get_time_difference(published)))
                row_count += 1
        else:  # asset_type == "crypto"
            for article in articles:
                table.insertRow(row_count)
                published = dt.datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%S.%fZ")
                table.setItem(row_count, 0,
                              QtWidgets.QTableWidgetItem(f"{article['title']} - {article['source']['name']}"))
                table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(self.get_time_difference(published)))
                row_count += 1

    @staticmethod
    def get_time_difference(published_time):
        published_now_difference = (dt.datetime.now() - published_time)
        difference_days = published_now_difference.days
        difference_hours = published_now_difference.seconds // 3600
        difference_minutes = (published_now_difference.seconds // 60) % 60

        if difference_days > 0:
            return f"{difference_days}d ago"
        elif (difference_days == 0) and (difference_hours > 0):
            return f"{difference_hours}h {difference_minutes}m ago"
        else:
            return f"{difference_minutes}m ago"

    @staticmethod
    def populate_table(table, data_type):
        row_count = 0
        latest_data = get_table_data(data_type)

        for ticker in latest_data:
            table.insertRow(row_count)
            column_count = 0
            for data in ticker:
                if column_count == 0 or column_count == 1:
                    table.setItem(row_count, column_count, QtWidgets.QTableWidgetItem(data))
                else:
                    table.setItem(row_count, column_count, QNumericalTableWidgetItem(str(data)))
                column_count += 1
            row_count += 1

        for column_count in range(table.columnCount()):
            for row_count in range(table.rowCount()):
                table.item(row_count, column_count).setTextAlignment(QtCore.Qt.AlignCenter)

    def get_asset_data_from_yahoo(self, symbol, start, end):
        try:
            df = pdr.get_data_yahoo(symbol, start, end)
            return df
        except (RemoteDataError, Exception):
            QtWidgets.QMessageBox.about(self, "Error", f"{symbol} could not be found")
            return None

    def add_comparison_ticker(self):
        ticker = self.lineEditCompany.text().upper()
        self.lineEditCompany.setText("")
        start = self.dateEditStartDate.date().toPyDate()
        end = self.dateEditEndDate.date().toPyDate()

        if ticker in self.plotted_lines:
            QtWidgets.QMessageBox.about(self, "Error", f"{ticker} already plotted")
            return

        df = self.get_asset_data_from_yahoo(ticker, start, end)
        if df is None:
            return

        self.plotted_lines[ticker] = self.plot_lines(self.ax1, self.ax2, df, ticker)

        self.add_plot_to_plotted_lines_table(self.tableComparisonTickers, ticker)

        self.ax1.legend()

        if not self.checkBoxOpen.isChecked():
            self.change_open_line_visibility()

        if not self.checkBoxHigh.isChecked():
            self.change_high_line_visibility()

        if not self.checkBoxLow.isChecked():
            self.change_low_line_visibility()

        if not self.checkBoxClose.isChecked():
            self.change_close_line_visibility()

        if not self.checkBoxMovingAverage.isChecked():
            self.change_ma_line_visibility()

        if not self.checkBoxForecast.isChecked():
            self.change_forecast_line_visibility()

        if not self.checkBoxCandlestick.isChecked():
            self.change_candlestick_line_visibility()

        # Recompute data limits based on new plots
        self.ax1.relim()
        self.ax2.relim()
        # Rescale view limits to above data limits
        self.ax1.autoscale_view()
        self.ax2.autoscale_view()

        self.fig.canvas.draw()

    def add_comparison_crypto(self):
        crypto = self.lineEditCrypto.text().upper()
        self.lineEditCrypto.setText("")
        start = self.dateEditStartDateCrypto.date().toPyDate()
        end = self.dateEditEndDateCrypto.date().toPyDate()

        if crypto in self.plotted_lines_crypto:
            QtWidgets.QMessageBox.about(self, "Error", f"{crypto} already plotted")
            return

        df = self.get_asset_data_from_yahoo(crypto, start, end)
        if df is None:
            return

        self.plotted_lines_crypto[crypto] = self.plot_lines(self.ax1_crypto, self.ax2_crypto, df, crypto)

        self.add_plot_to_plotted_lines_table(self.tableComparisonCrypto, crypto)

        self.ax1_crypto.legend()

        if not self.checkBoxOpenCrypto.isChecked():
            self.change_open_line_visibility_crypto()

        if not self.checkBoxHighCrypto.isChecked():
            self.change_high_line_visibility_crypto()

        if not self.checkBoxLowCrypto.isChecked():
            self.change_low_line_visibility_crypto()

        if not self.checkBoxCloseCrypto.isChecked():
            self.change_close_line_visibility_crypto()

        if not self.checkBoxMovingAverageCrypto.isChecked():
            self.change_ma_line_visibility_crypto()

        if not self.checkBoxForecastCrypto.isChecked():
            self.change_forecast_line_visibility_crypto()

        if not self.checkBoxCandlestickCrypto.isChecked():
            self.change_candlestick_line_visibility_crypto()

        # Recompute data limits based on new plots
        self.ax1_crypto.relim()
        self.ax2_crypto.relim()
        # Rescale view limits to above data limits
        self.ax1_crypto.autoscale_view()
        self.ax2_crypto.autoscale_view()

        self.fig_crypto.canvas.draw()

    def plot_lines(self, axis1, axis2, asset_df, asset):
        asset_df['50ma'] = asset_df['Close'].rolling(window=50, min_periods=0).mean()
        df_ohlc = asset_df['Close'].resample('10D').ohlc()
        df_ohlc.reset_index(inplace=True)
        df_volume = asset_df['Volume'].resample('1D').sum()
        df_ohlc['Date'] = df_ohlc['Date'].map(mpl_dates.date2num)  # Convert datetime dates to mpl_dates

        rgb = QtWidgets.QColorDialog.getColor().getRgb()[:-1]
        rgb = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

        open_line = axis1.plot(asset_df['Open'], color=rgb, linestyle='dashed')
        high_line = axis1.plot(asset_df['High'], color=rgb, linestyle='dashdot')
        low_line = axis1.plot(asset_df['Low'], color=rgb, linestyle='dotted')
        close_line = axis1.plot(asset_df['Close'], color=rgb, linestyle='solid', label=asset)
        ma_line = axis1.plot(asset_df['50ma'], color=rgb, linestyle='solid')
        forecast_line = axis1.plot(forecast_data(asset)['Forecast'], color=rgb, linestyle='dashed')
        candlestick_line = candlestick_ohlc(axis1, df_ohlc.values, colorup='g')
        volume_line = axis2.plot(df_volume, color=rgb)

        return [open_line, high_line, low_line, close_line, ma_line, forecast_line,
                candlestick_line, volume_line, rgb]

    @staticmethod
    def add_plot_to_plotted_lines_table(comparison_table, asset):
        row_num = comparison_table.rowCount()
        comparison_table.insertRow(row_num)
        item = QtWidgets.QTableWidgetItem(asset)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        comparison_table.setItem(row_num, 0, item)

    def remove_comparison_ticker(self, item):
        ticker = item.data()
        lines_to_rem = self.plotted_lines[ticker]

        self.ax1.lines.remove(lines_to_rem[0][0])  # Open
        self.ax1.lines.remove(lines_to_rem[1][0])  # High
        self.ax1.lines.remove(lines_to_rem[2][0])  # Low
        self.ax1.lines.remove(lines_to_rem[3][0])  # Close
        self.ax1.lines.remove(lines_to_rem[4][0])  # MA
        self.ax1.lines.remove(lines_to_rem[5][0])  # Forecast

        for line in lines_to_rem[6][0]:  # Candlestick lines
            self.ax1.lines.remove(line)
        for patch in lines_to_rem[6][1]:  # Candlestick patches
            patch.remove()
        self.ax2.lines.remove(lines_to_rem[7][0])  # Volume

        if len(self.ax1.lines) > 0:
            # Recompute data limits based on remaining plots
            self.ax1.relim()
            self.ax2.relim()
            # Rescale view limits to above data limits
            self.ax1.autoscale_view()
            self.ax2.autoscale_view()

        self.ax1.legend()
        self.fig.canvas.draw()

        self.plotted_lines.pop(ticker, None)

        # Remove from table
        table_item = self.tableComparisonTickers.findItems(ticker, QtCore.Qt.MatchContains)
        table_item_row = table_item[0].row()
        self.tableComparisonTickers.removeRow(table_item_row)

    def remove_comparison_crypto(self, item):
        crypto = item.data()
        lines_to_rem = self.plotted_lines_crypto[crypto]

        self.ax1_crypto.lines.remove(lines_to_rem[0][0])  # Open
        self.ax1_crypto.lines.remove(lines_to_rem[1][0])  # High
        self.ax1_crypto.lines.remove(lines_to_rem[2][0])  # Low
        self.ax1_crypto.lines.remove(lines_to_rem[3][0])  # Close
        self.ax1_crypto.lines.remove(lines_to_rem[4][0])  # MA
        self.ax1_crypto.lines.remove(lines_to_rem[5][0])  # Forecast

        for line in lines_to_rem[6][0]:  # Candlestick lines
            self.ax1_crypto.lines.remove(line)
        for patch in lines_to_rem[6][1]:  # Candlestick patches
            patch.remove()
        self.ax2_crypto.lines.remove(lines_to_rem[7][0])  # Volume

        if len(self.ax1_crypto.lines) > 0:
            # Recompute data limits based on remaining plots
            self.ax1_crypto.relim()
            self.ax2_crypto.relim()
            # Rescale view limits to above data limits
            self.ax1_crypto.autoscale_view()
            self.ax2_crypto.autoscale_view()

        self.ax1_crypto.legend()
        self.fig_crypto.canvas.draw()

        self.plotted_lines_crypto.pop(crypto, None)

        # Remove from table
        table_item = self.tableComparisonCrypto.findItems(crypto, QtCore.Qt.MatchContains)
        table_item_row = table_item[0].row()
        self.tableComparisonCrypto.removeRow(table_item_row)

    def change_open_line_visibility(self):
        for ticker_line in self.plotted_lines:
            open_line = self.plotted_lines[ticker_line][0][0]
            open_line.set_visible(self.checkBoxOpen.isChecked())
        self.fig.canvas.draw()

    def change_high_line_visibility(self):
        for ticker_line in self.plotted_lines:
            high_line = self.plotted_lines[ticker_line][1][0]
            high_line.set_visible(self.checkBoxHigh.isChecked())
        self.fig.canvas.draw()

    def change_low_line_visibility(self):
        for ticker_line in self.plotted_lines:
            low_line = self.plotted_lines[ticker_line][2][0]
            low_line.set_visible(self.checkBoxLow.isChecked())
        self.fig.canvas.draw()

    def change_close_line_visibility(self):
        for ticker_line in self.plotted_lines:
            close_line = self.plotted_lines[ticker_line][3][0]
            close_line.set_visible(self.checkBoxClose.isChecked())
        self.fig.canvas.draw()

    def change_ma_line_visibility(self):
        for ticker_line in self.plotted_lines:
            ma_line = self.plotted_lines[ticker_line][4][0]
            ma_line.set_visible(self.checkBoxMovingAverage.isChecked())
        self.fig.canvas.draw()

    def change_forecast_line_visibility(self):
        for ticker_line in self.plotted_lines:
            forecast_line = self.plotted_lines[ticker_line][5][0]
            forecast_line.set_visible(self.checkBoxForecast.isChecked())
        self.fig.canvas.draw()

    def change_candlestick_line_visibility(self):
        for ticker_line in self.plotted_lines:
            candlestick_lines = self.plotted_lines[ticker_line][6][0]
            for line in candlestick_lines:
                line.set_visible(self.checkBoxCandlestick.isChecked())

            patches = self.plotted_lines[ticker_line][6][1]
            for patch in patches:
                patch.set_visible(self.checkBoxCandlestick.isChecked())
        self.fig.canvas.draw()

    def change_open_line_visibility_crypto(self):
        for crypto_line in self.plotted_lines_crypto:
            open_line = self.plotted_lines_crypto[crypto_line][0][0]
            open_line.set_visible(self.checkBoxOpenCrypto.isChecked())
        self.fig_crypto.canvas.draw()

    def change_high_line_visibility_crypto(self):
        for crypto_line in self.plotted_lines_crypto:
            high_line = self.plotted_lines_crypto[crypto_line][1][0]
            high_line.set_visible(self.checkBoxHighCrypto.isChecked())
        self.fig_crypto.canvas.draw()

    def change_low_line_visibility_crypto(self):
        for crypto_line in self.plotted_lines_crypto:
            low_line = self.plotted_lines_crypto[crypto_line][2][0]
            low_line.set_visible(self.checkBoxLowCrypto.isChecked())
        self.fig_crypto.canvas.draw()

    def change_close_line_visibility_crypto(self):
        for crypto_line in self.plotted_lines_crypto:
            close_line = self.plotted_lines_crypto[crypto_line][3][0]
            close_line.set_visible(self.checkBoxCloseCrypto.isChecked())
        self.fig_crypto.canvas.draw()

    def change_ma_line_visibility_crypto(self):
        for crypto_line in self.plotted_lines_crypto:
            ma_line = self.plotted_lines_crypto[crypto_line][4][0]
            ma_line.set_visible(self.checkBoxMovingAverageCrypto.isChecked())
        self.fig_crypto.canvas.draw()

    def change_forecast_line_visibility_crypto(self):
        for crypto_line in self.plotted_lines_crypto:
            forecast_line = self.plotted_lines_crypto[crypto_line][5][0]
            forecast_line.set_visible(self.checkBoxForecastCrypto.isChecked())
        self.fig_crypto.canvas.draw()

    def change_candlestick_line_visibility_crypto(self):
        for crypto_line in self.plotted_lines_crypto:
            candlestick_lines = self.plotted_lines_crypto[crypto_line][6][0]
            for line in candlestick_lines:
                line.set_visible(self.checkBoxCandlestickCrypto.isChecked())
            patches = self.plotted_lines_crypto[crypto_line][6][1]
            for patch in patches:
                patch.set_visible(self.checkBoxCandlestickCrypto.isChecked())
        self.fig_crypto.canvas.draw()

    def save_notes(self):
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save file")
        if name == "":
            return

        file = open(name, "w")
        text = self.textEditNotes.toPlainText()
        file.write(text)
        file.close()

    def open_notes(self):
        name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file to open")
        if name == "":
            return

        file = open(name, "r")
        with file:
            text = file.read()
            self.textEditNotes.setPlainText(text)
        file.close()


class QNumericalTableWidgetItem(QtWidgets.QTableWidgetItem):
    # Custom table item that allows sorting via numerical values

    def __init__(self, number):
        QtWidgets.QTableWidgetItem.__init__(self, number, QtWidgets.QTableWidgetItem.UserType)
        self.__number = number

    def __lt__(self, other):
        return float(self.__number) < float(other.__number)


class LogInWindow(QtWidgets.QDialog):
    def __init__(self):
        super(LogInWindow, self).__init__()
        uic.loadUi("ui/login.ui", self)
        self.show()

        self.pushButtonSubmit.clicked.connect(self.attempt_login)
        self.pushButtonRegister.clicked.connect(self.open_register_window)

    def attempt_login(self):
        email = self.lineEditEmail.text().lower()
        passwd = self.lineEditPassword.text().encode("utf-8")

        val = (email,)
        sql = "SELECT id, email, firstname, surname, password FROM accounts WHERE email=%s"
        DB_CURSOR.execute(sql, val)
        result = DB_CURSOR.fetchall()

        if len(result) == 1:
            fetched_passwd = result[0][4].encode("utf-8")
            if bcrypt.checkpw(passwd, fetched_passwd):
                global USER_ID, USER_EMAIL, USER_FIRSTNAME, USER_LASTNAME, USER_PASSWORD_LENGTH
                USER_ID = result[0][0]
                USER_EMAIL = result[0][1]
                USER_FIRSTNAME = result[0][2]
                USER_LASTNAME = result[0][3]
                USER_PASSWORD_LENGTH = len(passwd)
                self.close()
                MainWindow()
            else:
                QtWidgets.QMessageBox.warning(self, "Invalid login", "Incorrect password")
        elif len(result) > 1:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          "Duplicate email detected in database")
        else:
            QtWidgets.QMessageBox.warning(self, "Invalid login", f"Email '{email}' not found")

    def open_register_window(self):
        self.register_window = RegisterWindow()


class RegisterWindow(QtWidgets.QDialog):
    def __init__(self):
        super(RegisterWindow, self).__init__()
        uic.loadUi("ui/register.ui", self)
        self.show()

        self.pushButtonSubmit.clicked.connect(self.register_account)

    def register_account(self):
        email = self.lineEditEmail.text().lower()
        firstname = self.lineEditFirstname.text()
        surname = self.lineEditSurname.text()
        passwd1 = self.lineEditPassword1.text()
        passwd2 = self.lineEditPassword2.text()

        if self.validate_email(email) and firstname != "" and surname != "" and self.validate_password(passwd1,
                                                                                                       passwd2):
            hashed_passwd = bcrypt.hashpw(passwd1.encode("utf-8"), bcrypt.gensalt())

            sql = "INSERT INTO accounts (email, firstname, surname, password) VALUES (%s, %s, %s, %s)"
            val = (email, firstname, surname, hashed_passwd)
            DB_CURSOR.execute(sql, val)
            DB.commit()

            QtWidgets.QMessageBox.information(self, "Success", "Account created successfully!\nYou may now log in")
            self.close()

    def validate_email(self, email):
        sql = "SELECT email FROM accounts WHERE email=%s"
        val = (email,)
        DB_CURSOR.execute(sql, val)
        result = DB_CURSOR.fetchall()

        if len(result) > 0:
            QtWidgets.QMessageBox.warning(self, "Error", f"An account already exists under {email}")
            return False
        elif email.find("@") == -1:
            QtWidgets.QMessageBox.warning(self, "Error", f"{email} is an invalid email address")
            return False
        return True

    def validate_password(self, password1, password2):
        if password1 != password2:
            QtWidgets.QMessageBox.warning(self, "Error", "Passwords do not match!")
            return False
        return True


def get_table_data(data_type):
    if data_type == "stock":
        ftse_100_data = []
        for page_num in range(1, 7):
            if page_num == 1:
                ftse_100_df = pandas.read_html(
                    'https://www.londonstockexchange.com/exchange/prices-and-markets/stocks/indices/summary/summary-indices-constituents.html?index=UKX',
                    attrs={'class': 'table_dati'})
            else:
                ftse_100_df = pandas.read_html(
                    f'https://www.londonstockexchange.com/exchange/prices-and-markets/stocks/indices/summary/summary-indices-constituents.html?index=UKX&page={page_num}',
                    attrs={'class': 'table_dati'})
            ftse_100_values = ftse_100_df[0].drop(
                columns=['Cur', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9'], axis=1).values
            for index in range(len(ftse_100_values)):
                ftse_100_values[index][0] = ftse_100_values[index][0].strip('.').replace('.', '-') + ".L"

            for company in ftse_100_values:
                ftse_100_data.append(list(company))
        return ftse_100_data
    else:  # data_type == "crypto"
        crypto_data = pandas.read_html('https://coinmarketcap.com/all/views/all/')
        crypto_data = crypto_data[2].drop(columns=["#", "Circulating Supply", "Volume (24h)", "Unnamed: 10"], axis=1)

        cols = crypto_data.columns
        new_cols = [cols[1], cols[0], *list(cols[3:]), cols[2]]

        crypto_data = crypto_data[new_cols]
        crypto_values = crypto_data.values

        for index in range(len(crypto_values)):
            crypto_values[index][0] = crypto_values[index][0] + "-USD"
            crypto_values[index][2] = crypto_values[index][2].strip('$').replace(',', '')
            crypto_values[index][3] = crypto_values[index][3].strip('%').replace(',', '')
            crypto_values[index][4] = crypto_values[index][4].strip('%').replace(',', '')
            crypto_values[index][5] = crypto_values[index][5].strip('%').replace(',', '')
            crypto_values[index][6] = crypto_values[index][6].strip('$').replace(',', '')

        return crypto_values


def set_ftse_100_tickers():
    ftse100_data = pandas.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index', attrs={
        'id': 'constituents'})  # Get table from FTSE 100 Wiki page with id 'constituents'
    table = ftse100_data[0]
    table.drop(columns=["Company", "FTSE Industry Classification Benchmark sector[12]"],
               inplace=True)
    ticker_list = table['Ticker'].tolist()

    for i in range(len(ticker_list)):
        ticker_list[i] = ticker_list[i].strip('.')
        ticker_list[i] = ticker_list[i].replace('.', '-')
        ticker_list[i] += ".L"

    global FTSE_100_TICKERS
    FTSE_100_TICKERS = ticker_list


def set_crypto_tickers():
    currencies = ['GBP', 'EUR', 'AUD', 'JPY', 'CAD', 'CNY', 'INR', 'KRW', 'RUB']
    crypto_data = pandas.read_html('https://finance.yahoo.com/cryptocurrencies/?count=50&offset=0',
                                   attrs={'class': 'W(100%)'})
    crypto_df = crypto_data[0]

    usd_crypto_list = crypto_df['Symbol'].tolist()
    all_currencies_crypto_list = []

    for crypto in usd_crypto_list:
        for currency in currencies:
            all_currencies_crypto_list.append(crypto.replace('-USD', f'-{currency}'))
    all_currencies_crypto_list += usd_crypto_list

    global CRYPTO_TICKERS
    CRYPTO_TICKERS = all_currencies_crypto_list


def set_sp_500_tickers():
    sp500_data = pandas.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', attrs={
        'id': 'constituents'})
    table = sp500_data[0]
    ticker_list = [ticker.replace('.', '-') for ticker in table['Symbol'].tolist()]

    global SP_500_TICKERS
    SP_500_TICKERS = ticker_list


def forecast_data(ticker):
    df = pdr.get_data_yahoo(ticker, dt.datetime.now() - dt.timedelta(days=3652), dt.datetime.now())

    # Pct difference between high and low
    df["HL_Pct_Diff"] = (abs(df["High"] - df["Low"]) / ((df["High"] + df["Low"]) / 2)) * 100
    # Pct difference between open and close
    df["OC_Pct_Diff"] = (abs(df["Open"] - df["Close"]) / ((df["Open"] + df["Close"]) / 2)) * 100

    feature_df = df[["HL_Pct_Diff", "OC_Pct_Diff", "Close"]]
    predict_col = "Close"
    days_to_predict = 7

    df["label"] = df[predict_col].shift(-days_to_predict)
    label_df = df[["label"]]

    X = numpy.array(feature_df)
    X_recent = X[-days_to_predict:]
    X = X[:-days_to_predict]

    y = numpy.array(label_df["label"])
    y = y[:-days_to_predict]

    scalar = preprocessing.StandardScaler()
    linear_regression = LinearRegression()
    pipeline = Pipeline(steps=[('transformer', scalar), ('estimator', linear_regression)])
    pipeline.fit(X, y)
    forecast_set = pipeline.predict(X_recent)

    last_row = df.tail(1)

    df["Forecast"] = numpy.nan

    last_date = df.iloc[-1].name  # Get date of last day
    last_date_unix = dt.datetime.timestamp(last_date)
    one_day = 86400  # Seconds in a day
    next_unix_day = last_date_unix + one_day

    # Adds label data to data frame and sets corresponding feature rows to nan
    for forecast in forecast_set:
        next_date = dt.datetime.fromtimestamp(next_unix_day)
        df.loc[next_date] = [numpy.nan for _ in range(len(df.columns) - 1)] + [forecast]
        next_unix_day += one_day

    df_forecast = df["Forecast"].dropna().to_frame()
    last_row = last_row.drop(columns=["HL_Pct_Diff", "OC_Pct_Diff", "label"], axis=1)
    last_row.rename(columns={"Adj Close": "Forecast"}, inplace=True)

    return pandas.concat([last_row, df_forecast], axis=0, sort=True)


def run_gui():
    app = QtWidgets.QApplication(sys.argv)
    login = LogInWindow()
    sys.exit(app.exec_())


# --=MAIN=--

run_gui()
