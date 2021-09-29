"""
TODO: dodac system powiadomiec

* nagly spadek akcji (o iles tam procent)




Learn Algorithmic Trading: Build and deploy algorithmic trading systems and ...
By Sebastien Donadio, Sourav Ghosh




"""
import warnings
import datetime
import logging

import numpy as np
import pandas as pd
from scipy.stats import invgamma

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import mplfinance as mpf


plt.style.use('fivethirtyeight')
st.set_option('deprecation.showPyplotGlobalUse', False)

stocks = [
    'AMD',
    'SIRI',
    'UCO',
    'USO',
    'EURPLN=X',
    'BTC-USD',
    'EOS-USD',
    'CAJ',
    'AAPL',
    'NEWR',
    'AMZN',
    'INTC',
    'TSLA',
    'TWTR',
    'NFLX',
    'SPOT',
    'GOOG',
    'FB',
    'GC=F', # GOLD
    'CL=F', # OIL
    '^GSPC', # SP500
    '^DJI', # DJ30
    'SI=F', # Silver
    'WIG.PA',
    'XOM',
]
def_stocks = ['AMD', 'SIRI', 'USO', 'UCO', 'CAJ', 'NEWR']



@st.cache(persist=True, show_spinner=False)
def extract(ticker, start_date=None, end_date=None):
    """ Fetch data
    """
    if start_date is None:
        start_date = datetime.datetime(2019, 5, 1)
    if end_date is None:
        end_date = datetime.datetime.now()

    ticker_ohlc = web.get_data_yahoo(ticker,
                             start=start_date,
                             end=end_date)
    # ticker_ohlc.info()
    # ticker_ohlc.Volume = ticker_ohlc.Volume.astype(float)
    return ticker_ohlc


def transform(ticker_ohlc, ticker):
    """ turn raw data into useful data in internal representation

    returns: pd.DataFrame
    """
    w_len = 5
    phi = 1.618
    centering = False

    df = pd.DataFrame({
        'ticker': ticker,
        f'Close': ticker_ohlc.Close,
        f'rolling-mean-3': ticker_ohlc.Close.rolling(window=3,center=centering).mean(),
        f'rolling-mean-5': ticker_ohlc.Close.rolling(window=5,center=centering).mean(),
        f'rolling-mean-21': ticker_ohlc.Close.rolling(window=21,center=centering).mean(),
        f'rolling-mean-50': ticker_ohlc.Close.rolling(window=50,center=centering).mean(),
        # f'rolling-mean-3': ticker_ohlc.Close.rolling(window=3,center=centering).mean(),
        # f'rolling-mean-{w_len}': ticker_ohlc.Close.rolling(window=w_len,center=centering).mean(),
        f'rolling-min-{w_len}': ticker_ohlc.Close.rolling(window=w_len,center=True).min(),
        f'rolling-max-{w_len}': ticker_ohlc.Close.rolling(window=w_len,center=True).max(),
    })
    return df


def plot_ticker(ticker, df=None, start_date=None, end_date=None):
    """ Plotting a ticker
    """
    if df is None:
        df = transform(extract(ticker, start_date, end_date), ticker)
    ax = df.plot(figsize=(12,5), lw=2, title=f'{ticker} Close')
    plt.legend(loc='upper left')
    # plt.ylim((ticker_ohlc.Close.min() * 0.9, ticker_ohlc.Close.max() * 1.1))
    return df



def plot_strategy(ticker, ticker_ac, short_window, mid_window):
    signals = pd.DataFrame(index=ticker_ac.index)
    signals['signal'] = 0.0
    roll_d10 = ticker_ac.rolling(window=short_window).mean()
    roll_d50 = ticker_ac.rolling(window=mid_window).mean()
    signals['short_mavg'] = roll_d10
    signals['mid_mavg'] = roll_d50
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['mid_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    plt.figure(figsize=(14, 7))
    plt.plot(ticker_ac.index, ticker_ac, lw=3, alpha=0.8,label='Original observations')
    plt.plot(ticker_ac.index, roll_d10, lw=3, alpha=0.8,label=f'Rolling mean (window {short_window})')
    plt.plot(ticker_ac.index, roll_d50, lw=3, alpha=0.8,label=f'Rolling mean (window {mid_window})')
    plt.plot(signals.loc[signals.positions == 1.0].index, 
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='r', label='buy')

    plt.plot(signals.loc[signals.positions == -1.0].index, 
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k', label='sell')
    plt.title(f'{ticker} Close Price (Technical Approach)')
    plt.tick_params(labelsize=12)
    plt.legend(loc='lower left', fontsize=12)
    plt.show()
    return


import functools
def plot_stock_correlations(stocks, startdd, endd):
	data = {}
	dfj = []
	for s in stocks:
	    data[s] = extract(ticker=s, start_date=startdd, end_date=endd)
	    _df = pd.DataFrame({f'[{s}]-Close': data[s].Close})
	    dfj.append(_df)

	ff = lambda a,b: a.join(b, how='outer')
	ddff = functools.reduce(ff, dfj)
	corr = ddff.corr()
	# Generate a mask for the upper triangle
	mask = np.tril(np.ones_like(corr, dtype=np.bool))

	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(12, 8))
	plt.title('Stock Correlations')
	plt.tight_layout()

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	# Draw the heatmap with the mask and correct aspect ratio
	_= sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
	            square=True, linewidths=.5, cbar_kws={"shrink": .9})
	return


def plot_historical(t0, t0_ohlc, span=5, linewidth=2):
	price = 'Close'
	lag = 1

	f, ax = plt.subplots(2, 1, figsize=(14,8), sharex=True)
	f.suptitle(f'{t0} Daily Returns')

	df = pd.DataFrame({
		f'{t0}-price':	t0_ohlc.loc[:, price],
		f'{t0}-lag{lag}': t0_ohlc.loc[:, price].shift(lag),
	})
	df['ratio'] = df.iloc[:, 0] / df.iloc[:, 1]
	df['return'] = df['ratio'].apply(np.log)
	df[f'return_ewm{span}'] = df['return'].ewm(span=span).mean()
	df[['return', f'return_ewm{span}']].plot(lw=linewidth, ax=ax[0])
	ax[0].axhline(y=0, color='k', lw=linewidth)

	df[f'EWMA{span}'] = df['return'].apply(lambda x: x*x).ewm(span=span).mean()
	df[f'EWMA{span}'].plot(lw=linewidth, ax=ax[1])
	return_var = df['return'].var()
	ax[1].axhline(y=return_var, color='r', lw=linewidth, label='var(return)')
	ax[1].axhline(y=0, color='k', lw=linewidth)
	plt.legend()

	#
	return


def render_investip():
	"""Render streamlit dashboard for investip
	"""
	linewidth = 2

	st.sidebar.markdown('# Dashboard')
	stock = st.sidebar.selectbox('Stock:', stocks)

	startdd = datetime.datetime(2020, 3, 1)
	startdd = st.sidebar.date_input('start-date', value=startdd)

	endd = datetime.datetime.now()
	endd = st.sidebar.date_input('end-date', value=endd)

	t0 = stock
	t0_ohlc = extract(ticker=t0, start_date=startdd, end_date=endd)
	t0_df = pd.DataFrame({f'{t0}-Close': t0_ohlc.Close})

	# st.write(t0_ohlc)
	mpf.plot(t0_ohlc, type='candle',volume=True,show_nontrading=False, title=t0, figscale=1.)
	# tdf = plot_ticker(t0, df=t0_df, start_date=startdd, end_date=endd)
	st.pyplot()


	st.sidebar.markdown('## Stock Correlation')
	stock_returns = st.sidebar.checkbox('Enable', value=True, key='cb_corrs')
	if stock_returns:
		st.markdown('## Stock Correlation')
		stock_selection = st.sidebar.multiselect('Stocks', stocks, def_stocks)
		plot_stock_correlations(stock_selection, startdd, endd)
		st.pyplot()

	# trading_context = True
	st.sidebar.markdown('## Returns')
	stock_returns = st.sidebar.checkbox('Enable', value=True, key='cb_returns')
	if stock_returns:
		st.markdown('## Stock Returns')
		st.markdown('''### Daily Stock returns
[EWMA](https://www.investopedia.com/articles/07/ewma.asp)''')
		span = st.sidebar.slider('span', 2, 21, value=5)
		plot_historical(t0, t0_ohlc, span=span, linewidth=linewidth)
		st.pyplot()


	# trading_context = True
	st.sidebar.markdown('## Volatility')
	trading_context = st.sidebar.checkbox('Enable', value=False, key='cb_volatility')
	if trading_context:
		st.markdown('## Volatility & Risk')
		st.markdown('''### Daily differences between High & Low
We model these ranges with [Inverse Gamma PDF](https://en.wikipedia.org/wiki/Inverse-gamma_distribution).
Green lines denote +/- 1 stdev.
''')
		f, ax = plt.subplots(1, 2, figsize=(14,6), sharex=False)
		f.suptitle(f'{t0} High-Low Daily')
		mmd = t0_ohlc.High - t0_ohlc.Low
		# mmd.dropna(inplace=True)
		mmd.plot(color='r', ax=ax[0], lw=linewidth)

		mu, sigma = mmd.dropna().mean(), mmd.dropna().std()
		zval = 1.#96
		# TODO: try one-tail limit to get outliers
		_=ax[0].axhline(y=mu, color='k', lw=linewidth)
		_=ax[0].axhline(y=mu-zval*sigma, color='g', lw=linewidth)
		_=ax[0].axhline(y=mu+zval*sigma, color='g', lw=linewidth)

		p95 = mmd.dropna().quantile(.95)
		_=ax[0].axhline(y=p95, color='b', lw=linewidth, label='p95')
		_=ax[1].axvline(p95, color='b', lw=linewidth, label='p95')

		with warnings.catch_warnings():
		    warnings.filterwarnings("ignore", category=RuntimeWarning)
		    print(invgamma.fit(mmd))
		    sns.distplot(mmd, fit=invgamma, kde=False, ax=ax[1])
		_=ax[1].axvline(mmd.values[-1], color='r', label='last', lw=linewidth)
		_=ax[1].axvline(mu, color='k', label='mean', lw=linewidth)
		_=ax[1].legend()
		st.pyplot()

		st.markdown('''### Daily Average True Range (ATR)
Implementation follows [ATR](https://kodify.net/tradingview/indicators/average-true-range/).
Check [Investopedia](https://www.investopedia.com/terms/a/atr.asp) for more info.''')

		atr_df = pd.DataFrame({
			f'{t0}-High-Low': t0_ohlc.High - t0_ohlc.Low,
			f'{t0}-High-PrevCloseAbs': abs(t0_ohlc.High - t0_ohlc.Close.shift(1)),
			f'{t0}-Low-PrevCloseAbs': abs(t0_ohlc.Low - t0_ohlc.Close.shift(1)),
		}).max(axis=1)
		atr_df = pd.DataFrame({
			f'{t0}-true-range': atr_df,
		})
		atr_df[f'{t0}-ATR14'] = atr_df.iloc[:, 0].rolling(14).mean()
		# st.write(atr_df)

		f, ax = plt.subplots(1, 2, figsize=(14,6), sharex=False)
		f.suptitle(f'{t0} True Range & SMA14')
		atr_df.plot(ax=ax[0], lw=linewidth)

		with warnings.catch_warnings():
		    warnings.filterwarnings("ignore", category=RuntimeWarning)
		    #print(invgamma.fit(f'{t0}-true-range'))
		    sns.distplot(atr_df[f'{t0}-true-range'], fit=invgamma, kde=False, ax=ax[1])
		_=ax[1].axvline(atr_df[f'{t0}-true-range'].values[-1], color='b', label='last', lw=linewidth)
		_=ax[1].axvline(atr_df[f'{t0}-ATR14'].values[-1], color='r', label='last', lw=linewidth)
		_=ax[1].legend()
		st.pyplot()



	# do_strategy_analysis = True
	st.sidebar.markdown('## Trading Strategy')
	do_strategy_analysis = st.sidebar.checkbox('Enable', value=False, key='cb_stra')
	if do_strategy_analysis:
		st.markdown('## Trading Strategy')
		st.markdown('[investopedia](https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp)')
		short_window = st.sidebar.slider('short_window', 2, 21, 3)
		long_window =  st.sidebar.slider('long_window', 3, 50, 5)
		plot_strategy(t0, t0_df, short_window, long_window)
		st.pyplot()

	# do_corr_analysis = False
	st.sidebar.markdown('## Correlation analysis')
	do_corr_analysis = st.sidebar.checkbox('Enable', value=False, key='cb_corr')
	if do_corr_analysis:
		st.markdown('## Correlation analysis')
		t1= 'GC=F' #  # SP500 'GC=F'
		t2 = 'CL=F' # '^GSPC' # '^DJI' # DJ30 'CL=F'
		t1 = st.sidebar.selectbox('REF1:', stocks, index=stocks.index(t1))
		t2 = st.sidebar.selectbox('REF2:', stocks, index=stocks.index(t2))
		if st.sidebar.button('Reset'):
			t1 = 'GC=F' #  # SP500 'GC=F'
			t2 = 'CL=F' # '^GSPC' # '^DJI' # DJ30 'CL=F'
			# t1 = st.sidebar.selectbox('ref1:', stocks, index=stocks.index(t1))
			# t2 = st.sidebar.selectbox('ref2:', stocks, index=stocks.index(t2))

		@st.cache(persist=True, show_spinner=False)
		def get_dataframes(t1, t2, startdd, endd):
			t1_ohlc = extract(ticker=t1, start_date=startdd, end_date=endd)
			t2_ohlc = extract(ticker=t2, start_date=startdd, end_date=endd)
			return t1_ohlc, t2_ohlc

		t1_ohlc, t2_ohlc = get_dataframes(t1, t2, startdd, endd)
		t1_df = pd.DataFrame({f'{t1}-Close': t1_ohlc.Close})
		t2_df = pd.DataFrame({f'{t2}-Close': t2_ohlc.Close})

		#print(t0_ohlc.shape)
		#t0_ohlc.head()
		# print(t1_ohlc.shape)
		# ticker_ohlc.head()
		# ticker_ohlc.info()

		tdf = t0_df.join(t1_df).join(t2_df).interpolate().dropna()
		# tdf.head(10)

		# t0_ohlc.corr(t1_ohlc)
		#ax = t0_ohlc.Close.plot()
		#t1_ohlc.Close.plot(ax=ax)

		import numpy as np
		print('glocal corrleation1: ', t0_ohlc.Close.corr(t1_ohlc.Close))
		print('glocal corrleation2: ', t0_ohlc.Close.corr(t2_ohlc.Close))

		p_window_size = 5
		r_window_size = 5
		centering = False


		modf = lambda x: x
		#modf = np.log10


		main_stat  = f'[{t0}]-mean-roll{p_window_size}'
		alt_stat_1 = f'[{t1}]-mean-roll{p_window_size}'
		alt_stat_2 = f'[{t2}]-mean-roll{p_window_size}'
		# df_rc = pd.DataFrame({
		#     main_stat : tdf.iloc[:, 0].apply(modf).rolling(window=p_window_size,center=centering).mean(),
		#     alt_stat_1: tdf.iloc[:, 1].apply(modf).rolling(window=p_window_size,center=centering).mean(),
		#     alt_stat_2: tdf.iloc[:, 2].apply(modf).rolling(window=p_window_size,center=centering).mean(),
		# })
		com_val = 0.2
		df_rc = pd.DataFrame({
		    main_stat : tdf.iloc[:, 0].apply(modf).ewm(span=p_window_size, adjust=False).mean(),
		    alt_stat_1: tdf.iloc[:, 1].apply(modf).ewm(span=p_window_size, adjust=False).mean(),
		    alt_stat_2: tdf.iloc[:, 2].apply(modf).ewm(span=p_window_size, adjust=False).mean(),
		})

		df_rc = df_rc.interpolate()
		df_rc[f'[{t0}]-[{t1}]-corr-roll{r_window_size}'] = df_rc[main_stat].rolling(window=r_window_size, center=centering).corr(df_rc[alt_stat_1])
		df_rc[f'[{t0}]-[{t2}]-corr-roll{r_window_size}'] = df_rc[main_stat].rolling(window=r_window_size, center=centering).corr(df_rc[alt_stat_2])

		f, ax = plt.subplots(3,1,figsize=(16,10),sharex=True)
		#df_rc.iloc[:,0].plot(ax=ax[0], legend=True)
		df_rc.iloc[:,1].plot(ax=ax[0], legend=True, color='gold')
		df_rc.iloc[:,2].plot(ax=ax[1], legend=True, color='darkred')
		df_rc.iloc[:,3].plot(ax=ax[2], legend=True, color='gold')
		df_rc.iloc[:,4].plot(ax=ax[2], legend=True, color='darkred')
		ax[2].axhline(y=0, lw=1, color='black')
		#t0_ohlc.Close.rolling(window=r_window_size,center=True).mean().plot(ax=ax[0])
		#t1_ohlc.Close.rolling(window=r_window_size,center=True).mean().plot(ax=ax[1])
		# ax[0].set(xlabel='Frame',ylabel='Smiling Evidence')
		# ax[1].set(xlabel='Frame',ylabel='Pearson r')
		_=plt.suptitle(f"{t0} Close rolling correlation to {t1}, {t2}")

		st.pyplot()


		f,ax=plt.subplots(1, 2, figsize=(16,8),sharex=False)

		_= df_rc.plot.scatter(x=df_rc.columns[1],
		                      y=df_rc.columns[2],
		                      c=df_rc.columns[0],
		                      colormap='viridis',
		                      # legend=None,
		                      ax=ax[0])

		print(df_rc.columns)
		newr_p = df_rc.iloc[-1, 0]
		t1_p = df_rc.iloc[-1, 1]
		t2_p = df_rc.iloc[-1, 2]
		t1_c = df_rc.dropna().iloc[-1, 3]
		t2_c = df_rc.dropna().iloc[-1, 4]
		print('current_corr:', (t1_c, t2_c))

		# figure out circle size
		aaaa = df_rc.iloc[:, 1].aggregate([np.max, np.min])
		xrange = np.ceil(aaaa.values[0] - aaaa.values[1])
		print(aaaa.values[0], aaaa.values[1], xrange)
		xradius = xrange / 20.

		circle = plt.Circle((t1_p, t2_p), xradius, color='r', fill=False)
		ax[0].add_artist(circle)
		#ax[0].set_xlabel(f'GOLD Price {t1_p:.4f}')
		#ax[0].set_ylabel(f'OIL Price {t2_p:.4f}')
		# ax[0].legend().set_visible(False)

		_= df_rc.plot.scatter(x=df_rc.columns[-2],
		                      y=df_rc.columns[-1],
		                      c=df_rc.columns[0],
		                      colormap='viridis',
		                      # legend=True,
		                      #linestyle=
		                      ax=ax[1])

		# figure out circle size
		aaaa = df_rc.iloc[:, -2].aggregate([np.max, np.min])
		xrange = np.ceil(aaaa.values[0] - aaaa.values[1])
		print(aaaa.values[0], aaaa.values[1], xrange)
		xradius = xrange / 20.

		circle1 = plt.Circle((t1_c, t2_c), xradius, color='r', fill=False)
		ax[1].add_artist(circle1)
		#ax[1].set_ylabel('OIL Correlation')
		#_= ax[1].set_xlabel('GOLD Correlation')


		st.pyplot()


render_investip()

