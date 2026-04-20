#!/usr/bin/env python
# coding: utf-8



# Install required packages


# Step 2: Define State Structure
# Create a shared state class that all agents can read from and write to:
# Stock symbol and basic info
# Market data (prices, volumes, indicators)
# Sentiment scores and news data
# Technical analysis results
# Risk metrics
# Final recommendation
# Step 3: Create Individual Agent Functions
# 3.1 Data Analysis Agent
# Function to fetch real-time stock data (yfinance, Alpha Vantage)
# Function to get historical price data
# Function to fetch financial statements
# Function to get market indices data
# Function to collect economic indicators
# 3.2 Sentiment Analysis Agent
# Function to fetch recent news articles
# Function to analyze news sentiment using NLP
# Function to get social media sentiment (Twitter API/Reddit)
# Function to collect analyst recommendations
# Function to calculate overall sentiment score
# 3.3 Technical Analysis Agent
# Function to calculate moving averages (SMA, EMA)
# Function to compute momentum indicators (RSI, MACD, Stochastic)
# Function to identify support/resistance levels
# Function to detect chart patterns
# Function to analyze volume trends
# 3.4 Risk Assessment Agent
# Function to calculate Value at Risk (VaR)
# Function to compute Beta and correlation
# Function to assess volatility metrics
# Function to analyze portfolio impact
# Function to determine risk score
# 3.5 Decision Making Agent
# Function to synthesize all analyses
# Function to apply decision criteria
# Function to generate recommendations
# Function to set price targets and stop losses
# Function to calculate confidence scores

# Cell 1-2: Environment Setup and Imports
# 



# Cell 1: Install required packages (run this first)
"""
pip install langgraph langchain-openai yfinance alpha-vantage newsapi-python
pip install pandas numpy scikit-learn textblob requests beautifulsoup4
pip install ta plotly matplotlib seaborn fredapi tweepy praw kiteconnect
"""

# Cell 2: Import all required libraries
import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Data manipulation and analysis
import pandas as pd
import numpy as np
from scipy import stats
import ta

# Financial data sources
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import fredapi

# News and sentiment analysis
import requests
from newsapi import NewsApiClient
import tweepy
import praw
from textblob import TextBlob
from bs4 import BeautifulSoup

# Trading execution
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None
    logging.warning("kiteconnect not installed. Run: pip install kiteconnect")

# LangGraph and LangChain
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# Visualization
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ### Cell 3-4: Define State Classes and Data Structures
# 



# Cell 3: Define the main state structure

import operator

# Reducers for LangGraph parallel branch merging
def _last(a, b):
    """Last-write-wins: safe for fields owned by a single node."""
    return b

def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts; values from b overwrite a."""
    return {**a, **b}

# Annotated[T, reducer] tells LangGraph how to merge values when two parallel
# nodes both return a value for the same key.
class StockAnalysisState(TypedDict):
    # Input parameters
    symbol:               Annotated[str,            _last]
    company_name:         Annotated[str,            _last]
    analysis_date:        Annotated[datetime,       _last]

    # Market data
    current_price:        Annotated[float,          _last]
    price_change:         Annotated[float,          _last]
    price_change_percent: Annotated[float,          _last]
    volume:               Annotated[int,            _last]
    market_cap:           Annotated[float,          _last]

    # Historical data
    historical_data:      Annotated[pd.DataFrame,   _last]
    financial_data:       Annotated[Dict[str, Any], _last]

    # Sentiment analysis results
    news_sentiment:       Annotated[float,          _last]
    social_sentiment:     Annotated[float,          _last]
    analyst_consensus:    Annotated[float,          _last]
    overall_sentiment:    Annotated[float,          _last]
    sentiment_details:    Annotated[Dict[str, Any], _last]

    # Technical analysis results
    technical_indicators: Annotated[Dict[str, float], _last]
    support_resistance:   Annotated[Dict[str, float], _last]
    trend_direction:      Annotated[str,              _last]
    chart_patterns:       Annotated[List[str],        _last]

    # Risk assessment results
    risk_score:           Annotated[float,            _last]
    volatility:           Annotated[float,            _last]
    beta:                 Annotated[float,            _last]
    var_score:            Annotated[float,            _last]
    risk_metrics:         Annotated[Dict[str, float], _last]

    # Final recommendation
    recommendation:       Annotated[str,            _last]
    confidence_score:     Annotated[float,          _last]
    price_target:         Annotated[float,          _last]
    stop_loss:            Annotated[float,          _last]
    rationale:            Annotated[str,            _last]

    # Trade execution results
    trade_executed:       Annotated[bool,             _last]
    trade_details:        Annotated[Dict[str, Any],   _last]
    order_id:             Annotated[Optional[str],    _last]

    # Execution tracking — lists use add (append), dicts use merge
    errors:               Annotated[List[str],          operator.add]
    warnings:             Annotated[List[str],          operator.add]
    execution_time:       Annotated[Dict[str, float],   _merge_dicts]


# Cell 4: Define configuration and helper classes
@dataclass
class APIConfig:
    """Configuration for various APIs"""
    alpha_vantage_key: str = ""
    news_api_key: str = ""
    fred_api_key: str = ""
    twitter_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    openai_api_key: str = ""
    kite_api_key: str = ""
    kite_access_token: str = ""

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    lookback_days: int = 252
    short_ma_period: int = 20
    long_ma_period: int = 50
    rsi_period: int = 14
    bollinger_period: int = 20
    confidence_threshold: float = 0.6
    risk_free_rate: float = 0.02

@dataclass
class TradingConfig:
    """Configuration for trade execution via Kite Connect"""
    paper_trading: bool = True          # Set False to place real orders
    confidence_threshold: float = 80.0  # Only trade when confidence >= this
    exchange: str = "NSE"
    default_quantity: int = 1
    max_order_value: float = 50000.0    # Max order value in INR
    order_type: str = "MARKET"          # MARKET or LIMIT
    product: str = "CNC"                # CNC = delivery, MIS = intraday

@dataclass
class TechnicalIndicators:
    """Structure for technical indicators"""
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_middle: float = 0.0
    atr: float = 0.0
    volume_sma: float = 0.0


# ### Cell 5-9: Implement Data Analysis Agent Functions



# Cell 5: Core data fetching functions
class DataAnalysisAgent:
    def __init__(self, config: APIConfig):
        self.config = config
        self.alpha_vantage = TimeSeries(key=config.alpha_vantage_key) if config.alpha_vantage_key else None
        self.fundamental_data = FundamentalData(key=config.alpha_vantage_key) if config.alpha_vantage_key else None
        self.fred = fredapi.Fred(api_key=config.fred_api_key) if config.fred_api_key else None
    
    def fetch_stock_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Fetch current stock data and basic info"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period=period)
            
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            previous_close = info.get('previousClose', hist['Close'].iloc[-2])
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'current_price': current_price,
                'previous_close': previous_close,
                'price_change': current_price - previous_close,
                'price_change_percent': ((current_price - previous_close) / previous_close) * 100,
                'volume': info.get('volume', hist['Volume'].iloc[-1]),
                'market_cap': info.get('marketCap', 0),
                'historical_data': hist,
                'info': info
            }
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return {}

# Cell 6: Historical data analysis
    def get_historical_analysis(self, historical_data: pd.DataFrame, days: int = 252) -> Dict[str, float]:
        """Analyze historical performance metrics"""
        try:
            if len(historical_data) < days:
                days = len(historical_data)
            
            recent_data = historical_data.tail(days)
            returns = recent_data['Close'].pct_change().dropna()
            
            return {
                'avg_return': returns.mean() * 252,  # Annualized
                'volatility': returns.std() * np.sqrt(252),  # Annualized
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(recent_data['Close']),
                'avg_volume': recent_data['Volume'].mean(),
                'price_range_52w': {
                    'high': recent_data['High'].max(),
                    'low': recent_data['Low'].min()
                }
            }
        except Exception as e:
            logger.error(f"Error in historical analysis: {e}")
            return {}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

# Cell 7: Financial statements analysis
    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch and analyze financial statements"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get financial statements
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Key financial ratios
            info = ticker.info
            financial_ratios = {
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0)
            }
            
            return {
                'income_statement': income_stmt.to_dict() if not income_stmt.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {},
                'financial_ratios': financial_ratios,
                'analyst_info': {
                    'target_high_price': info.get('targetHighPrice', 0),
                    'target_low_price': info.get('targetLowPrice', 0),
                    'target_mean_price': info.get('targetMeanPrice', 0),
                    'recommendation_mean': info.get('recommendationMean', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error fetching financial data: {e}")
            return {}

# Cell 8: Market indices and sector analysis
    def get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get market and sector context"""
        try:
            # Major indices
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^VIX': 'VIX'
            }
            
            market_data = {}
            for idx_symbol, name in indices.items():
                try:
                    idx_ticker = yf.Ticker(idx_symbol)
                    idx_hist = idx_ticker.history(period="5d")
                    if not idx_hist.empty:
                        current = idx_hist['Close'].iloc[-1]
                        previous = idx_hist['Close'].iloc[-2]
                        change_pct = ((current - previous) / previous) * 100
                        market_data[name] = {
                            'current': current,
                            'change_percent': change_pct
                        }
                except:
                    continue
            
            # Get sector information
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector_info = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            return {
                'market_indices': market_data,
                'sector_info': sector_info
            }
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {}

# Cell 9: Economic indicators
    def get_economic_indicators(self) -> Dict[str, float]:
        """Fetch key economic indicators"""
        try:
            if not self.fred:
                return {}
            
            indicators = {
                'GDP': 'GDP',
                'UNRATE': 'Unemployment Rate',
                'CPIAUCSL': 'CPI',
                'FEDFUNDS': 'Federal Funds Rate',
                'DGS10': '10-Year Treasury Rate'
            }
            
            economic_data = {}
            for code, name in indicators.items():
                try:
                    data = self.fred.get_series(code, limit=1)
                    if not data.empty:
                        economic_data[name] = data.iloc[-1]
                except:
                    continue
            
            return economic_data
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            return {}


# ### Cell 10-14: Implement Sentiment Analysis Agent Functions



# Cell 10: News sentiment analysis
class SentimentAnalysisAgent:
    def __init__(self, config: APIConfig):
        self.config = config
        self.news_api = NewsApiClient(api_key=config.news_api_key) if config.news_api_key else None
        self.setup_social_media_clients()
    
    def setup_social_media_clients(self):
        """Setup social media API clients"""
        # Twitter client
        if self.config.twitter_bearer_token:
            self.twitter_client = tweepy.Client(bearer_token=self.config.twitter_bearer_token)
        else:
            self.twitter_client = None
        
        # Reddit client
        if self.config.reddit_client_id and self.config.reddit_client_secret:
            self.reddit_client = praw.Reddit(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                user_agent="StockAnalyzer"
            )
        else:
            self.reddit_client = None
    
    def analyze_news_sentiment(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Analyze news sentiment for a stock"""
        try:
            company_name = yf.Ticker(symbol).info.get('longName', symbol)
            
            # Fetch news articles
            articles = self._fetch_news_articles(symbol, company_name, days_back)
            
            if not articles:
                return {'sentiment_score': 0.0, 'article_count': 0, 'articles': []}
            
            # Analyze sentiment of each article
            sentiments = []
            analyzed_articles = []
            
            for article in articles:
                title_sentiment = TextBlob(article['title']).sentiment.polarity
                description_sentiment = TextBlob(article.get('description', '')).sentiment.polarity
                
                # Weight title more heavily
                combined_sentiment = (title_sentiment * 0.7) + (description_sentiment * 0.3)
                sentiments.append(combined_sentiment)
                
                analyzed_articles.append({
                    'title': article['title'],
                    'sentiment': combined_sentiment,
                    'published_at': article['publishedAt'],
                    'source': article['source']['name']
                })
            
            # Calculate overall sentiment
            overall_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            return {
                'sentiment_score': overall_sentiment,
                'article_count': len(articles),
                'articles': analyzed_articles[:10],  # Top 10 articles
                'sentiment_distribution': {
                    'positive': len([s for s in sentiments if s > 0.1]),
                    'neutral': len([s for s in sentiments if -0.1 <= s <= 0.1]),
                    'negative': len([s for s in sentiments if s < -0.1])
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {'sentiment_score': 0.0, 'article_count': 0, 'articles': []}

# Cell 11: News fetching helper
    def _fetch_news_articles(self, symbol: str, company_name: str, days_back: int) -> List[Dict]:
        """Fetch news articles from various sources"""
        articles = []
        
        # NewsAPI
        if self.news_api:
            try:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                # Search by symbol and company name
                queries = [symbol, company_name]
                
                for query in queries:
                    news_response = self.news_api.get_everything(
                        q=query,
                        from_param=from_date,
                        language='en',
                        sort_by='relevancy',
                        page_size=50
                    )
                    
                    if news_response['status'] == 'ok':
                        articles.extend(news_response['articles'])
            except Exception as e:
                logger.error(f"NewsAPI error: {e}")
        
        # Yahoo Finance news (backup)
        try:
            ticker = yf.Ticker(symbol)
            yahoo_news = ticker.news
            for item in yahoo_news[:20]:  # Limit to 20 articles
                articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'publishedAt': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'source': {'name': item.get('publisher', 'Yahoo Finance')}
                })
        except Exception as e:
            logger.error(f"Yahoo Finance news error: {e}")
        
        # Remove duplicates and sort by date
        unique_articles = {article['title']: article for article in articles}.values()
        return sorted(unique_articles, key=lambda x: x['publishedAt'], reverse=True)

# Cell 12: Social media sentiment
    def analyze_social_sentiment(self, symbol: str, days_back: int = 3) -> Dict[str, Any]:
        """Analyze social media sentiment"""
        twitter_sentiment = self._get_twitter_sentiment(symbol, days_back)
        reddit_sentiment = self._get_reddit_sentiment(symbol, days_back)
        
        # Combine sentiments
        sentiments = []
        sources = []
        
        if twitter_sentiment['sentiment_score'] != 0:
            sentiments.append(twitter_sentiment['sentiment_score'])
            sources.append('Twitter')
        
        if reddit_sentiment['sentiment_score'] != 0:
            sentiments.append(reddit_sentiment['sentiment_score'])
            sources.append('Reddit')
        
        overall_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        return {
            'sentiment_score': overall_sentiment,
            'twitter': twitter_sentiment,
            'reddit': reddit_sentiment,
            'sources_analyzed': sources,
            'total_mentions': twitter_sentiment.get('tweet_count', 0) + reddit_sentiment.get('post_count', 0)
        }
    
    def _get_twitter_sentiment(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get Twitter sentiment for a stock"""
        try:
            if not self.twitter_client:
                return {'sentiment_score': 0.0, 'tweet_count': 0}
            
            # Search for tweets
            query = f"${symbol} OR {symbol} -is:retweet lang:en"
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            ).flatten(limit=200)
            
            sentiments = []
            for tweet in tweets:
                sentiment = TextBlob(tweet.text).sentiment.polarity
                sentiments.append(sentiment)
            
            return {
                'sentiment_score': np.mean(sentiments) if sentiments else 0.0,
                'tweet_count': len(sentiments)
            }
        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
            return {'sentiment_score': 0.0, 'tweet_count': 0}
    
    def _get_reddit_sentiment(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get Reddit sentiment for a stock"""
        try:
            if not self.reddit_client:
                return {'sentiment_score': 0.0, 'post_count': 0}
            
            # Search in relevant subreddits
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'StockMarket']
            all_sentiments = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    for post in subreddit.search(symbol, limit=50):
                        # Analyze post title and content
                        text = f"{post.title} {post.selftext}"
                        sentiment = TextBlob(text).sentiment.polarity
                        all_sentiments.append(sentiment)
                except:
                    continue
            
            return {
                'sentiment_score': np.mean(all_sentiments) if all_sentiments else 0.0,
                'post_count': len(all_sentiments)
            }
        except Exception as e:
            logger.error(f"Reddit sentiment error: {e}")
            return {'sentiment_score': 0.0, 'post_count': 0}

# Cell 13: Analyst sentiment
    def get_analyst_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get analyst recommendations and sentiment"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            recommendations = ticker.recommendations
            
            # Get recommendation summary
            recommendation_mean = info.get('recommendationMean', 3.0)  # 1=Strong Buy, 5=Strong Sell
            
            # Convert to sentiment score (-1 to 1)
            # 1-2: Strong positive, 2-3: Positive, 3-4: Negative, 4-5: Strong negative
            sentiment_score = (5 - recommendation_mean) / 2 - 1  # Convert to -1 to 1 scale
            
            analyst_data = {
                'sentiment_score': max(-1, min(1, sentiment_score)),
                'recommendation_mean': recommendation_mean,
                'target_price': info.get('targetMeanPrice', 0),
                'number_of_analysts': info.get('numberOfAnalystOpinions', 0)
            }
            
            # Get recent recommendations if available
            if recommendations is not None and not recommendations.empty:
                recent_recs = recommendations.head(10)
                analyst_data['recent_recommendations'] = recent_recs.to_dict('records')
            
            return analyst_data
        except Exception as e:
            logger.error(f"Error getting analyst sentiment: {e}")
            return {'sentiment_score': 0.0, 'recommendation_mean': 3.0}

# Cell 14: Overall sentiment calculation
    def calculate_overall_sentiment(self, news_sentiment: float, social_sentiment: float, 
                                  analyst_sentiment: float) -> Dict[str, Any]:
        """Calculate weighted overall sentiment"""
        try:
            # Weights for different sentiment sources
            weights = {
                'news': 0.4,
                'social': 0.3,
                'analyst': 0.3
            }
            
            # Calculate weighted average
            overall_sentiment = (
                news_sentiment * weights['news'] +
                social_sentiment * weights['social'] +
                analyst_sentiment * weights['analyst']
            )
            
            # Classify sentiment
            if overall_sentiment > 0.3:
                sentiment_label = "Bullish"
            elif overall_sentiment > 0.1:
                sentiment_label = "Slightly Bullish"
            elif overall_sentiment > -0.1:
                sentiment_label = "Neutral"
            elif overall_sentiment > -0.3:
                sentiment_label = "Slightly Bearish"
            else:
                sentiment_label = "Bearish"
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_label': sentiment_label,
                'component_scores': {
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'analyst': analyst_sentiment
                },
                'weights_used': weights
            }
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': "Neutral",
                'component_scores': {},
                'weights_used': {}
            }


# ### Cell 15-19: Implement Technical Analysis Agent Functions
# 



# Cell 15: Technical Analysis Agent class
class TechnicalAnalysisAgent:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators"""
        try:
            if df.empty or len(df) < self.config.long_ma_period:
                return {}
            
            indicators = {}
            
            # Moving Averages
            indicators.update(self._calculate_moving_averages(df))
            
            # Momentum Indicators
            indicators.update(self._calculate_momentum_indicators(df))
            
            # Volatility Indicators
            indicators.update(self._calculate_volatility_indicators(df))
            
            # Volume Indicators
            indicators.update(self._calculate_volume_indicators(df))
            
            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(df))
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various moving averages"""
        try:
            close = df['Close']
            
            return {
                'sma_20': ta.trend.sma_indicator(close, window=20).iloc[-1],
                'sma_50': ta.trend.sma_indicator(close, window=50).iloc[-1],
                'sma_200': ta.trend.sma_indicator(close, window=200).iloc[-1] if len(df) >= 200 else 0,
                'ema_12': ta.trend.ema_indicator(close, window=12).iloc[-1],
                'ema_26': ta.trend.ema_indicator(close, window=26).iloc[-1],
                'ema_50': ta.trend.ema_indicator(close, window=50).iloc[-1],
                'wma_20': ta.trend.wma_indicator(close, window=20).iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}

# Cell 16: Momentum indicators
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum-based indicators"""
        try:
            high, low, close, volume = df['High'], df['Low'], df['Close'], df['Volume']
            
            # RSI
            rsi = ta.momentum.rsi(close, window=self.config.rsi_period)
            
            # MACD
            macd_line = ta.trend.macd(close)
            macd_signal = ta.trend.macd_signal(close)
            macd_histogram = ta.trend.macd_diff(close)
            
            # Stochastic
            stoch_k = ta.momentum.stoch(high, low, close)
            stoch_d = ta.momentum.stoch_signal(high, low, close)
            
            # Williams %R
            williams_r = ta.momentum.williams_r(high, low, close)
            
            # Commodity Channel Index
            cci = ta.trend.cci(high, low, close)
            
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
                'macd_signal': macd_signal.iloc[-1] if not macd_signal.empty else 0,
                'macd_histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else 0,
                'stoch_k': stoch_k.iloc[-1] if not stoch_k.empty else 50,
                'stoch_d': stoch_d.iloc[-1] if not stoch_d.empty else 50,
                'williams_r': williams_r.iloc[-1] if not williams_r.empty else -50,
                'cci': cci.iloc[-1] if not cci.empty else 0
            }
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-based indicators"""
        try:
            high, low, close = df['High'], df['Low'], df['Close']
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(close, window=self.config.bollinger_period)
            bb_low = ta.volatility.bollinger_lband(close, window=self.config.bollinger_period)
            bb_mid = ta.volatility.bollinger_mavg(close, window=self.config.bollinger_period)
            bb_width = ta.volatility.bollinger_wband(close, window=self.config.bollinger_period)
            
            # Average True Range
            atr = ta.volatility.average_true_range(high, low, close)
            
            # Keltner Channel
            kc_high = ta.volatility.keltner_channel_hband(high, low, close)
            kc_low = ta.volatility.keltner_channel_lband(high, low, close)
            
            return {
                'bollinger_upper': bb_high.iloc[-1] if not bb_high.empty else close.iloc[-1],
                'bollinger_lower': bb_low.iloc[-1] if not bb_low.empty else close.iloc[-1],
                'bollinger_middle': bb_mid.iloc[-1] if not bb_mid.empty else close.iloc[-1],
                'bollinger_width': bb_width.iloc[-1] if not bb_width.empty else 0,
                'atr': atr.iloc[-1] if not atr.empty else 0,
                'keltner_upper': kc_high.iloc[-1] if not kc_high.empty else close.iloc[-1],
                'keltner_lower': kc_low.iloc[-1] if not kc_low.empty else close.iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return {}

# Cell 17: Volume and trend indicators
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        try:
            high, low, close, volume = df['High'], df['Low'], df['Close'], df['Volume']
            
            # On Balance Volume
            obv = ta.volume.on_balance_volume(close, volume)
            
            # Volume Weighted Average Price
            vwap = ta.volume.volume_weighted_average_price(high, low, close, volume)
            
            # Chaikin Money Flow
            cmf = ta.volume.chaikin_money_flow(high, low, close, volume)
            
            # Volume SMA
            volume_sma = ta.trend.sma_indicator(volume, window=20)
            
            return {
                'obv': obv.iloc[-1] if not obv.empty else 0,
                'vwap': vwap.iloc[-1] if not vwap.empty else close.iloc[-1],
                'cmf': cmf.iloc[-1] if not cmf.empty else 0,
                'volume_sma': volume_sma.iloc[-1] if not volume_sma.empty else volume.iloc[-1],
                'volume_ratio': volume.iloc[-1] / volume_sma.iloc[-1] if not volume_sma.empty and volume_sma.iloc[-1] > 0 else 1
            }
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend-based indicators"""
        try:
            high, low, close = df['High'], df['Low'], df['Close']
            
            # ADX (Average Directional Index)
            adx = ta.trend.adx(high, low, close)
            adx_pos = ta.trend.adx_pos(high, low, close)
            adx_neg = ta.trend.adx_neg(high, low, close)
            
            # Parabolic SAR
            psar = ta.trend.psar_down(high, low, close)
            
            # Aroon
            aroon_up = ta.trend.aroon_up(close)
            aroon_down = ta.trend.aroon_down(close)
            
            return {
                'adx': adx.iloc[-1] if not adx.empty else 25,
                'adx_pos': adx_pos.iloc[-1] if not adx_pos.empty else 25,
                'adx_neg': adx_neg.iloc[-1] if not adx_neg.empty else 25,
                'psar': psar.iloc[-1] if not psar.empty else close.iloc[-1],
                'aroon_up': aroon_up.iloc[-1] if not aroon_up.empty else 50,
                'aroon_down': aroon_down.iloc[-1] if not aroon_down.empty else 50
            }
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return {}

# Cell 18: Support and resistance analysis
    def identify_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """Identify support and resistance levels"""
        try:
            high, low, close = df['High'], df['Low'], df['Close']
            current_price = close.iloc[-1]
            
            # Calculate pivot points
            recent_data = df.tail(window * 3)  # Use more data for better accuracy
            
            # Find local maxima and minima
            highs = recent_data['High'].rolling(window=window, center=True).max()
            lows = recent_data['Low'].rolling(window=window, center=True).min()
            
            # Identify resistance levels (local maxima)
            resistance_candidates = []
            for i in range(window, len(recent_data) - window):
                if recent_data['High'].iloc[i] == highs.iloc[i]:
                    resistance_candidates.append(recent_data['High'].iloc[i])
            
            # Identify support levels (local minima)
            support_candidates = []
            for i in range(window, len(recent_data) - window):
                if recent_data['Low'].iloc[i] == lows.iloc[i]:
                    support_candidates.append(recent_data['Low'].iloc[i])
            
            # Find the closest support and resistance levels
            resistance_above = [r for r in resistance_candidates if r > current_price]
            support_below = [s for s in support_candidates if s < current_price]
            
            nearest_resistance = min(resistance_above) if resistance_above else current_price * 1.05
            nearest_support = max(support_below) if support_below else current_price * 0.95
            
            # Calculate additional levels using different methods
            # Fibonacci retracements
            recent_high = df['High'].tail(252).max()  # 1 year high
            recent_low = df['Low'].tail(252).min()    # 1 year low
            
            fib_levels = self._calculate_fibonacci_levels(recent_high, recent_low)
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_strength': len([s for s in support_candidates if abs(s - nearest_support) < current_price * 0.02]),
                'resistance_strength': len([r for r in resistance_candidates if abs(r - nearest_resistance) < current_price * 0.02]),
                'fibonacci_levels': fib_levels,
                'pivot_point': (recent_data['High'].iloc[-1] + recent_data['Low'].iloc[-1] + recent_data['Close'].iloc[-1]) / 3
            }
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return {}
    
    def _calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            'fib_0': high,
            'fib_23.6': high - (diff * 0.236),
            'fib_38.2': high - (diff * 0.382),
            'fib_50': high - (diff * 0.5),
            'fib_61.8': high - (diff * 0.618),
            'fib_100': low
        }

# Cell 19: Chart pattern recognition and trend analysis
    def analyze_trend_and_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend direction and identify chart patterns"""
        try:
            close = df['Close']
            current_price = close.iloc[-1]
            
            # Trend analysis using multiple timeframes
            trends = {
                'short_term': self._determine_trend(df.tail(20)),  # 20 days
                'medium_term': self._determine_trend(df.tail(50)), # 50 days
                'long_term': self._determine_trend(df.tail(200)) if len(df) >= 200 else 'Neutral'  # 200 days
            }
            
            # Overall trend (weighted)
            trend_scores = {
                'Upward': 1,
                'Sideways': 0,
                'Downward': -1,
                'Neutral': 0
            }
            
            weighted_trend = (
                trend_scores.get(trends['short_term'], 0) * 0.5 +
                trend_scores.get(trends['medium_term'], 0) * 0.3 +
                trend_scores.get(trends['long_term'], 0) * 0.2
            )
            
            if weighted_trend > 0.3:
                overall_trend = 'Upward'
            elif weighted_trend < -0.3:
                overall_trend = 'Downward'
            else:
                overall_trend = 'Sideways'
            
            # Pattern recognition
            patterns = self._identify_chart_patterns(df)
            
            # Momentum analysis
            momentum = self._analyze_momentum(df)
            
            return {
                'trends': trends,
                'overall_trend': overall_trend,
                'trend_strength': abs(weighted_trend),
                'patterns': patterns,
                'momentum': momentum,
                'trend_score': weighted_trend
            }
        except Exception as e:
            logger.error(f"Error in trend and pattern analysis: {e}")
            return {}
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine trend direction for given timeframe"""
        try:
            if len(df) < 10:
                return 'Neutral'
            
            close = df['Close']
            sma_short = close.rolling(window=min(10, len(df)//2)).mean()
            sma_long = close.rolling(window=min(20, len(df))).mean()
            
            current_price = close.iloc[-1]
            sma_short_current = sma_short.iloc[-1]
            sma_long_current = sma_long.iloc[-1]
            
            # Price vs moving averages
            if current_price > sma_short_current > sma_long_current:
                return 'Upward'
            elif current_price < sma_short_current < sma_long_current:
                return 'Downward'
            else:
                # Check slope of moving averages
                if len(sma_long) >= 5:
                    sma_slope = (sma_long.iloc[-1] - sma_long.iloc[-5]) / sma_long.iloc[-5]
                    if sma_slope > 0.02:
                        return 'Upward'
                    elif sma_slope < -0.02:
                        return 'Downward'
                
                return 'Sideways'
        except:
            return 'Neutral'
    
    def _identify_chart_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify basic chart patterns"""
        patterns = []
        
        try:
            if len(df) < 50:
                return patterns
            
            close = df['Close'].tail(50)
            high = df['High'].tail(50)
            low = df['Low'].tail(50)
            
            # Double top/bottom patterns
            if self._is_double_top(high):
                patterns.append('Double Top')
            
            if self._is_double_bottom(low):
                patterns.append('Double Bottom')
            
            # Head and shoulders
            if self._is_head_and_shoulders(high):
                patterns.append('Head and Shoulders')
            
            # Triangle patterns
            triangle_pattern = self._identify_triangle(high, low)
            if triangle_pattern:
                patterns.append(triangle_pattern)
            
            # Flag and pennant
            flag_pattern = self._identify_flag_pennant(close, high, low)
            if flag_pattern:
                patterns.append(flag_pattern)
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
        
        return patterns
    
    def _is_double_top(self, high: pd.Series) -> bool:
        """Identify double top pattern"""
        try:
            peaks = []
            for i in range(5, len(high) - 5):
                if high.iloc[i] == high.iloc[i-5:i+6].max():
                    peaks.append((i, high.iloc[i]))
            
            if len(peaks) >= 2:
                # Check if two highest peaks are similar in height
                peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                peak1, peak2 = peaks_sorted[0], peaks_sorted[1]
                height_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                return height_diff < 0.03  # Within 3%
            
            return False
        except:
            return False
    
    def _is_double_bottom(self, low: pd.Series) -> bool:
        """Identify double bottom pattern"""
        try:
            troughs = []
            for i in range(5, len(low) - 5):
                if low.iloc[i] == low.iloc[i-5:i+6].min():
                    troughs.append((i, low.iloc[i]))
            
            if len(troughs) >= 2:
                # Check if two lowest troughs are similar in depth
                troughs_sorted = sorted(troughs, key=lambda x: x[1])
                trough1, trough2 = troughs_sorted[0], troughs_sorted[1]
                depth_diff = abs(trough1[1] - trough2[1]) / trough1[1]
                return depth_diff < 0.03  # Within 3%
            
            return False
        except:
            return False
    
    def _is_head_and_shoulders(self, high: pd.Series) -> bool:
        """Identify head and shoulders pattern"""
        try:
            peaks = []
            for i in range(10, len(high) - 10):
                if high.iloc[i] == high.iloc[i-10:i+11].max():
                    peaks.append((i, high.iloc[i]))
            
            if len(peaks) >= 3:
                # Sort by height to find head (highest) and shoulders
                peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                head = peaks_sorted[0]
                potential_shoulders = peaks_sorted[1:3]
                
                # Check if shoulders are similar height and head is significantly higher
                if len(potential_shoulders) >= 2:
                    shoulder_diff = abs(potential_shoulders[0][1] - potential_shoulders[1][1]) / potential_shoulders[0][1]
                    head_shoulder_diff = (head[1] - potential_shoulders[0][1]) / potential_shoulders[0][1]
                    
                    return shoulder_diff < 0.05 and head_shoulder_diff > 0.1
            
            return False
        except:
            return False
    
    def _identify_triangle(self, high: pd.Series, low: pd.Series) -> Optional[str]:
        """Identify triangle patterns"""
        try:
            # Simple triangle identification based on converging trend lines
            recent_highs = high.tail(20)
            recent_lows = low.tail(20)
            
            # Calculate trend lines
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # Ascending triangle: horizontal resistance, rising support
            if abs(high_slope) < 0.01 and low_slope > 0.01:
                return 'Ascending Triangle'
            
            # Descending triangle: falling resistance, horizontal support
            elif high_slope < -0.01 and abs(low_slope) < 0.01:
                return 'Descending Triangle'
            
            # Symmetrical triangle: converging lines
            elif high_slope < -0.01 and low_slope > 0.01:
                return 'Symmetrical Triangle'
            
            return None
        except:
            return None
    
    def _identify_flag_pennant(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Optional[str]:
        """Identify flag and pennant patterns"""
        try:
            # Look for strong move followed by consolidation
            recent_data = close.tail(30)
            
            # Check for strong initial move (first 10 periods)
            initial_move = recent_data.iloc[:10]
            consolidation = recent_data.iloc[10:]
            
            initial_change = (initial_move.iloc[-1] - initial_move.iloc[0]) / initial_move.iloc[0]
            consolidation_volatility = consolidation.std() / consolidation.mean()
            
            # Strong move (>5%) followed by low volatility consolidation
            if abs(initial_change) > 0.05 and consolidation_volatility < 0.02:
                if initial_change > 0:
                    return 'Bull Flag'
                else:
                    return 'Bear Flag'
            
            return None
        except:
            return None
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price momentum"""
        try:
            close = df['Close']
            
            # Rate of change over different periods
            roc_5 = ((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100 if len(close) > 5 else 0
            roc_10 = ((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11]) * 100 if len(close) > 10 else 0
            roc_20 = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100 if len(close) > 20 else 0
            
            # Price momentum score
            momentum_scores = [roc_5 * 0.5, roc_10 * 0.3, roc_20 * 0.2]
            momentum_score = sum(momentum_scores)
            
            # Momentum classification
            if momentum_score > 5:
                momentum_strength = 'Strong Positive'
            elif momentum_score > 2:
                momentum_strength = 'Moderate Positive'
            elif momentum_score > -2:
                momentum_strength = 'Neutral'
            elif momentum_score > -5:
                momentum_strength = 'Moderate Negative'
            else:
                momentum_strength = 'Strong Negative'
            
            return {
                'momentum_score': momentum_score,
                'momentum_strength': momentum_strength,
                'roc_5d': roc_5,
                'roc_10d': roc_10,
                'roc_20d': roc_20
            }
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {}
        


# ### Cell 20-24: Implement Risk Assessment Agent Functions



# Cell 20: Risk Assessment Agent class
class RiskAssessmentAgent:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.risk_free_rate = config.risk_free_rate
    
    def calculate_comprehensive_risk(self, df: pd.DataFrame, symbol: str, 
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            risk_metrics = {}
            
            # Price volatility risk
            volatility_risk = self._calculate_volatility_risk(df)
            risk_metrics.update(volatility_risk)
            
            # Market risk (Beta)
            market_risk = self._calculate_market_risk(df, symbol)
            risk_metrics.update(market_risk)
            
            # Value at Risk
            var_metrics = self._calculate_var(df)
            risk_metrics.update(var_metrics)
            
            # Liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(df, market_data)
            risk_metrics.update(liquidity_risk)
            
            # Fundamental risk
            fundamental_risk = self._calculate_fundamental_risk(market_data)
            risk_metrics.update(fundamental_risk)
            
            # Overall risk score
            overall_risk = self._calculate_overall_risk_score(risk_metrics)
            risk_metrics['overall_risk_score'] = overall_risk
            
            return risk_metrics
        except Exception as e:
            logger.error(f"Error calculating comprehensive risk: {e}")
            return {}
    
    def _calculate_volatility_risk(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-based risk metrics"""
        try:
            close = df['Close']
            returns = close.pct_change().dropna()
            
            # Historical volatility (annualized)
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            # Volatility percentiles
            vol_30d = returns.tail(30).std() * np.sqrt(252)
            vol_90d = returns.tail(90).std() * np.sqrt(252) if len(returns) >= 90 else annual_vol
            
            # Volatility risk score (0-1, where 1 is highest risk)
            # Compare to typical stock volatility (15-25%)
            vol_risk_score = min(annual_vol / 0.4, 1.0)  # Cap at 40% volatility
            
            return {
                'annual_volatility': annual_vol,
                'volatility_30d': vol_30d,
                'volatility_90d': vol_90d,
                'volatility_risk_score': vol_risk_score,
                'volatility_percentile': self._calculate_volatility_percentile(returns)
            }
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {e}")
            return {}
    
    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """Calculate current volatility percentile vs historical"""
        try:
            if len(returns) < 60:
                return 0.5
            
            # Rolling 30-day volatilities
            rolling_vols = returns.rolling(30).std() * np.sqrt(252)
            current_vol = rolling_vols.iloc[-1]
            
            # Percentile of current volatility
            percentile = stats.percentileofscore(rolling_vols.dropna(), current_vol) / 100
            return percentile
        except:
            return 0.5

# Cell 21: Market risk calculation
    def _calculate_market_risk(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate market risk metrics including Beta"""
        try:
            # Get S&P 500 data for beta calculation
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")
            
            # Align dates
            common_dates = df.index.intersection(spy_hist.index)
            if len(common_dates) < 30:  # Need at least 30 data points
                return {'beta': 1.0, 'correlation_with_market': 0.0}
            
            stock_returns = df.loc[common_dates]['Close'].pct_change().dropna()
            market_returns = spy_hist.loc[common_dates]['Close'].pct_change().dropna()
            
            # Align returns
            common_dates_returns = stock_returns.index.intersection(market_returns.index)
            stock_returns = stock_returns.loc[common_dates_returns]
            market_returns = market_returns.loc[common_dates_returns]
            
            # Calculate beta
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            # Calculate correlation
            correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
            
            # Calculate alpha (excess return)
            stock_mean_return = stock_returns.mean()
            market_mean_return = market_returns.mean()
            alpha = stock_mean_return - (self.risk_free_rate/252 + beta * (market_mean_return - self.risk_free_rate/252))
            
            # Market risk score
            market_risk_score = min(abs(beta - 1) + (1 - abs(correlation)), 1.0)
            
            return {
                'beta': beta,
                'alpha': alpha * 252,  # Annualized
                'correlation_with_market': correlation,
                'market_risk_score': market_risk_score,
                'systematic_risk': beta * market_returns.std() * np.sqrt(252)
            }
        except Exception as e:
            logger.error(f"Error calculating market risk: {e}")
            return {'beta': 1.0, 'correlation_with_market': 0.0, 'market_risk_score': 0.5}

# Cell 22: Value at Risk calculation
    def _calculate_var(self, df: pd.DataFrame, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """Calculate Value at Risk using multiple methods"""
        try:
            close = df['Close']
            returns = close.pct_change().dropna()
            
            if len(returns) < 30:
                return {}
            
            var_metrics = {}
            
            # Historical VaR
            for confidence in confidence_levels:
                var_percentile = (1 - confidence) * 100
                historical_var = np.percentile(returns, var_percentile)
                var_metrics[f'var_historical_{int(confidence*100)}'] = historical_var
            
            # Parametric VaR (assuming normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            
            for confidence in confidence_levels:
                z_score = stats.norm.ppf(1 - confidence)
                parametric_var = mean_return + z_score * std_return
                var_metrics[f'var_parametric_{int(confidence*100)}'] = parametric_var
            
            # Monte Carlo VaR
            mc_var = self._monte_carlo_var(returns, confidence_levels)
            var_metrics.update(mc_var)
            
            # Expected Shortfall (CVaR)
            for confidence in confidence_levels:
                var_threshold = var_metrics.get(f'var_historical_{int(confidence*100)}', 0)
                expected_shortfall = returns[returns <= var_threshold].mean()
                var_metrics[f'expected_shortfall_{int(confidence*100)}'] = expected_shortfall
            
            # Maximum Drawdown
            var_metrics['max_drawdown'] = self._calculate_max_drawdown(close)
            
            return var_metrics
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {}
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_levels: List[float], 
                        simulations: int = 10000) -> Dict[str, float]:
        """Calculate VaR using Monte Carlo simulation"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, simulations)
            
            mc_var = {}
            for confidence in confidence_levels:
                var_percentile = (1 - confidence) * 100
                mc_var_value = np.percentile(simulated_returns, var_percentile)
                mc_var[f'var_monte_carlo_{int(confidence*100)}'] = mc_var_value
            
            return mc_var
        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR: {e}")
            return {}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except:
            return 0.0

# Cell 23: Liquidity and fundamental risk
    def _calculate_liquidity_risk(self, df: pd.DataFrame, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate liquidity risk metrics"""
        try:
            volume = df['Volume']
            close = df['Close']
            
            # Average daily volume
            avg_volume = volume.mean()
            
            # Volume volatility
            volume_volatility = volume.std() / avg_volume if avg_volume > 0 else 1
            
            # Bid-ask spread proxy (using high-low spread)
            spread_proxy = ((df['High'] - df['Low']) / df['Close']).mean()
            
            # Market cap from market_data
            market_cap = market_data.get('market_cap', 0)
            
            # Liquidity risk score
            if market_cap > 10e9:  # Large cap
                size_risk = 0.1
            elif market_cap > 2e9:  # Mid cap
                size_risk = 0.3
            else:  # Small cap
                size_risk = 0.6
            
            # Volume risk (lower volume = higher risk)
            volume_risk = min(1.0, 1000000 / avg_volume) if avg_volume > 0 else 1.0
            
            liquidity_risk_score = (size_risk + volume_risk + min(spread_proxy * 10, 1.0)) / 3
            
            return {
                'avg_volume': avg_volume,
                'volume_volatility': volume_volatility,
                'spread_proxy': spread_proxy,
                'liquidity_risk_score': liquidity_risk_score,
                'market_cap_risk': size_risk
            }
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return {}
    
    def _calculate_fundamental_risk(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fundamental risk based on financial metrics"""
        try:
            financial_ratios = market_data.get('financial_data', {}).get('financial_ratios', {})
            
            if not financial_ratios:
                return {'fundamental_risk_score': 0.5}
            
            risk_factors = {}
            
            # P/E ratio risk
            pe_ratio = financial_ratios.get('pe_ratio', 15)
            if pe_ratio > 30 or pe_ratio < 5:
                risk_factors['pe_risk'] = 0.8
            elif pe_ratio > 25 or pe_ratio < 8:
                risk_factors['pe_risk'] = 0.5
            else:
                risk_factors['pe_risk'] = 0.2
            
            # Debt-to-equity risk
            debt_to_equity = financial_ratios.get('debt_to_equity', 0)
            if debt_to_equity > 2:
                risk_factors['debt_risk'] = 0.9
            elif debt_to_equity > 1:
                risk_factors['debt_risk'] = 0.6
            else:
                risk_factors['debt_risk'] = 0.3
            
            # Profitability risk
            profit_margin = financial_ratios.get('profit_margin', 0)
            if profit_margin < 0:
                risk_factors['profitability_risk'] = 0.9
            elif profit_margin < 0.05:
                risk_factors['profitability_risk'] = 0.6
            else:
                risk_factors['profitability_risk'] = 0.2
            
            # Current ratio (liquidity)
            current_ratio = financial_ratios.get('current_ratio', 1)
            if current_ratio < 1:
                risk_factors['liquidity_risk'] = 0.8
            elif current_ratio < 1.5:
                risk_factors['liquidity_risk'] = 0.4
            else:
                risk_factors['liquidity_risk'] = 0.1
            
            # Calculate overall fundamental risk
            fundamental_risk_score = np.mean(list(risk_factors.values())) if risk_factors else 0.5
            
            return {
                'fundamental_risk_score': fundamental_risk_score,
                'risk_factors': risk_factors
            }
        except Exception as e:
            logger.error(f"Error calculating fundamental risk: {e}")
            return {'fundamental_risk_score': 0.5}

# Cell 24: Overall risk score calculation
    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall risk score and classification"""
        try:
            # Weight different risk components
            weights = {
                'volatility': 0.3,
                'market': 0.2,
                'liquidity': 0.2,
                'fundamental': 0.2,
                'var': 0.1
            }
            
            # Extract risk scores
            volatility_risk = risk_metrics.get('volatility_risk_score', 0.5)
            market_risk = risk_metrics.get('market_risk_score', 0.5)
            liquidity_risk = risk_metrics.get('liquidity_risk_score', 0.5)
            fundamental_risk = risk_metrics.get('fundamental_risk_score', 0.5)
            
            # VaR risk (convert to 0-1 scale)
            var_95 = abs(risk_metrics.get('var_historical_95', -0.02))
            var_risk = min(var_95 / 0.1, 1.0)  # Normalize to 10% daily loss as maximum
            
            # Calculate weighted overall risk
            overall_risk = (
                volatility_risk * weights['volatility'] +
                market_risk * weights['market'] +
                liquidity_risk * weights['liquidity'] +
                fundamental_risk * weights['fundamental'] +
                var_risk * weights['var']
            )
            
            # Risk classification
            if overall_risk < 0.3:
                risk_category = 'Low'
            elif overall_risk < 0.6:
                risk_category = 'Medium'
            else:
                risk_category = 'High'
            
            return {
                'overall_risk_score': overall_risk,
                'risk_category': risk_category,
                'risk_components': {
                    'volatility_risk': volatility_risk,
                    'market_risk': market_risk,
                    'liquidity_risk': liquidity_risk,
                    'fundamental_risk': fundamental_risk,
                    'var_risk': var_risk
                },
                'weights_used': weights
            }
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return {'overall_risk_score': 0.5, 'risk_category': 'Medium'}


# ### Cell 25-29: Implement Decision Making Agent Functions



# Cell 25: Decision Making Agent class
class DecisionMakingAgent:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.confidence_threshold = config.confidence_threshold
    
    def make_investment_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make final investment recommendation based on all analyses"""
        try:
            # Extract analysis results
            sentiment_data = self._extract_sentiment_data(state)
            technical_data = self._extract_technical_data(state)
            risk_data = self._extract_risk_data(state)
            fundamental_data = self._extract_fundamental_data(state)
            
            # Calculate individual recommendation scores
            sentiment_score = self._calculate_sentiment_score(sentiment_data)
            technical_score = self._calculate_technical_score(technical_data)
            risk_score = self._calculate_risk_score(risk_data)
            fundamental_score = self._calculate_fundamental_score(fundamental_data)
            
            # Weight and combine scores
            weighted_score = self._combine_scores(
                sentiment_score, technical_score, risk_score, fundamental_score
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(weighted_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                sentiment_score, technical_score, risk_score, fundamental_score, weighted_score
            )
            
            # Set price targets and stop loss
            price_targets = self._calculate_price_targets(state, recommendation, technical_data, risk_data)
            
            # Generate rationale
            rationale = self._generate_rationale(
                recommendation, sentiment_data, technical_data, risk_data, 
                fundamental_data, confidence
            )
            
            return {
                'recommendation': recommendation,
                'confidence_score': confidence,
                'weighted_score': weighted_score,
                'component_scores': {
                    'sentiment': sentiment_score,
                    'technical': technical_score,
                    'risk': risk_score,
                    'fundamental': fundamental_score
                },
                'price_target': price_targets['target'],
                'stop_loss': price_targets['stop_loss'],
                'upside_potential': price_targets['upside_percent'],
                'downside_risk': price_targets['downside_percent'],
                'risk_reward_ratio': price_targets['risk_reward_ratio'],
                'rationale': rationale,
                'time_horizon': self._determine_time_horizon(technical_data, recommendation)
            }
        except Exception as e:
            logger.error(f"Error making investment decision: {e}")
            return self._default_decision()
    
    def _extract_sentiment_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment analysis data from state"""
        return {
            'overall_sentiment': state.get('overall_sentiment', 0.0),
            'news_sentiment': state.get('news_sentiment', 0.0),
            'social_sentiment': state.get('social_sentiment', 0.0),
            'analyst_consensus': state.get('analyst_consensus', 0.0),
            'sentiment_details': state.get('sentiment_details', {})
        }
    
    def _extract_technical_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical analysis data from state"""
        return {
            'technical_indicators': state.get('technical_indicators', {}),
            'trend_direction': state.get('trend_direction', 'Neutral'),
            'support_resistance': state.get('support_resistance', {}),
            'chart_patterns': state.get('chart_patterns', [])
        }
    
    def _extract_risk_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk assessment data from state"""
        return {
            'risk_score': state.get('risk_score', 0.5),
            'volatility': state.get('volatility', 0.2),
            'beta': state.get('beta', 1.0),
            'var_score': state.get('var_score', -0.02),
            'risk_metrics': state.get('risk_metrics', {})
        }
    
    def _extract_fundamental_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fundamental analysis data from state"""
        return {
            'financial_data': state.get('financial_data', {}),
            'market_cap': state.get('market_cap', 0),
            'pe_ratio': state.get('financial_data', {}).get('financial_ratios', {}).get('pe_ratio', 15)
        }

# Cell 26: Score calculation methods
    def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate sentiment-based recommendation score (-1 to 1)"""
        try:
            overall_sentiment = sentiment_data.get('overall_sentiment', 0.0)
            
            # Normalize sentiment to -1 to 1 scale
            sentiment_score = max(-1, min(1, overall_sentiment))
            
            # Apply sentiment strength weighting
            news_sentiment = sentiment_data.get('news_sentiment', 0.0)
            social_sentiment = sentiment_data.get('social_sentiment', 0.0)
            analyst_sentiment = sentiment_data.get('analyst_consensus', 0.0)
            
            # Check for sentiment agreement (higher weight if all sources agree)
            sentiments = [news_sentiment, social_sentiment, analyst_sentiment]
            sentiment_std = np.std([s for s in sentiments if s != 0])
            
            # Lower standard deviation means more agreement
            agreement_bonus = max(0, (0.5 - sentiment_std) / 0.5) * 0.2 if sentiment_std > 0 else 0
            
            final_score = sentiment_score + (agreement_bonus if sentiment_score > 0 else -agreement_bonus)
            return max(-1, min(1, final_score))
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return 0.0
    
    def _calculate_technical_score(self, technical_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score (-1 to 1)"""
        try:
            indicators = technical_data.get('technical_indicators', {})
            trend = technical_data.get('trend_direction', 'Neutral')
            patterns = technical_data.get('chart_patterns', [])
            
            score_components = []
            
            # Trend score
            trend_scores = {'Upward': 0.6, 'Sideways': 0.0, 'Downward': -0.6, 'Neutral': 0.0}
            trend_score = trend_scores.get(trend, 0.0)
            score_components.append(trend_score)
            
            # RSI score
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                rsi_score = -0.3  # Overbought
            elif rsi < 30:
                rsi_score = 0.3   # Oversold
            else:
                rsi_score = (50 - abs(rsi - 50)) / 100  # Closer to 50 is neutral
            score_components.append(rsi_score)
            
            # MACD score
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_score = 0.2 if macd > macd_signal else -0.2
            score_components.append(macd_score)
            
            # Moving average score
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            current_price = indicators.get('current_price', sma_20)
            
            ma_score = 0
            if current_price > sma_20 > sma_50:
                ma_score = 0.4
            elif current_price < sma_20 < sma_50:
                ma_score = -0.4
            score_components.append(ma_score)
            
            # Pattern score
            pattern_score = 0
            bullish_patterns = ['Double Bottom', 'Bull Flag', 'Ascending Triangle']
            bearish_patterns = ['Double Top', 'Bear Flag', 'Descending Triangle', 'Head and Shoulders']
            
            for pattern in patterns:
                if pattern in bullish_patterns:
                    pattern_score += 0.2
                elif pattern in bearish_patterns:
                    pattern_score -= 0.2
            
            score_components.append(pattern_score)
            
            # Calculate weighted average
            final_score = np.mean(score_components)
            return max(-1, min(1, final_score))
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.0
    
    def _calculate_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate risk-adjusted score (-1 to 1, where negative = high risk)"""
        try:
            overall_risk = risk_data.get('risk_score', 0.5)
            volatility = risk_data.get('volatility', 0.2)
            beta = risk_data.get('beta', 1.0)
            
            # Convert risk to score (lower risk = higher score)
            risk_score = 1 - (overall_risk * 2)  # Convert 0-1 risk to 1 to -1 score
            
            # Adjust for volatility preference
            vol_adjustment = max(-0.3, min(0.1, (0.25 - volatility) / 0.25))
            
            # Beta adjustment (prefer moderate beta)
            beta_adjustment = max(-0.2, min(0.1, (1.5 - abs(beta - 1)) / 1.5))
            
            final_score = risk_score + vol_adjustment + beta_adjustment
            return max(-1, min(1, final_score))
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.0
    
    def _calculate_fundamental_score(self, fundamental_data: Dict[str, Any]) -> float:
        """Calculate fundamental analysis score (-1 to 1)"""
        try:
            financial_data = fundamental_data.get('financial_data', {})
            ratios = financial_data.get('financial_ratios', {})
            
            if not ratios:
                return 0.0
            
            score_components = []
            
            # P/E ratio score
            pe_ratio = ratios.get('pe_ratio', 15)
            if 10 <= pe_ratio <= 20:
                pe_score = 0.3
            elif 8 <= pe_ratio <= 25:
                pe_score = 0.1
            elif pe_ratio > 30:
                pe_score = -0.3
            else:
                pe_score = 0.0
            score_components.append(pe_score)
            
            # ROE score
            roe = ratios.get('roe', 0)
            if roe > 0.15:
                roe_score = 0.3
            elif roe > 0.10:
                roe_score = 0.1
            elif roe < 0:
                roe_score = -0.4
            else:
                roe_score = 0.0
            score_components.append(roe_score)
            
            # Debt-to-equity score
            debt_equity = ratios.get('debt_to_equity', 0)
            if debt_equity < 0.5:
                debt_score = 0.2
            elif debt_equity < 1.0:
                debt_score = 0.1
            elif debt_equity > 2.0:
                debt_score = -0.3
            else:
                debt_score = -0.1
            score_components.append(debt_score)
            
            # Profit margin score
            profit_margin = ratios.get('profit_margin', 0)
            if profit_margin > 0.2:
                margin_score = 0.3
            elif profit_margin > 0.1:
                margin_score = 0.2
            elif profit_margin > 0.05:
                margin_score = 0.1
            elif profit_margin < 0:
                margin_score = -0.4
            else:
                margin_score = 0.0
            score_components.append(margin_score)
            
            final_score = np.mean(score_components)
            return max(-1, min(1, final_score))
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 0.0

# Cell 27: Score combination and recommendation generation
    def _combine_scores(self, sentiment_score: float, technical_score: float, 
                       risk_score: float, fundamental_score: float) -> float:
        """Combine individual scores with weights"""
        try:
            # Define weights based on investment strategy
            weights = {
                'sentiment': 0.2,
                'technical': 0.3,
                'risk': 0.25,
                'fundamental': 0.25
            }
            
            weighted_score = (
                sentiment_score * weights['sentiment'] +
                technical_score * weights['technical'] +
                risk_score * weights['risk'] +
                fundamental_score * weights['fundamental']
            )
            
            return max(-1, min(1, weighted_score))
        except Exception as e:
            logger.error(f"Error combining scores: {e}")
            return 0.0
    
    def _generate_recommendation(self, weighted_score: float) -> str:
        """Generate recommendation based on weighted score"""
        try:
            if weighted_score > 0.6:
                return 'STRONG BUY'
            elif weighted_score > 0.2:
                return 'BUY'
            elif weighted_score > -0.2:
                return 'HOLD'
            elif weighted_score > -0.6:
                return 'SELL'
            else:
                return 'STRONG SELL'
        except:
            return 'HOLD'
    
    def _calculate_confidence(self, sentiment_score: float, technical_score: float,
                            risk_score: float, fundamental_score: float, 
                            weighted_score: float) -> float:
        """Calculate confidence score based on agreement between analyses"""
        try:
            scores = [sentiment_score, technical_score, risk_score, fundamental_score]
            
            # Calculate agreement (lower standard deviation = higher confidence)
            score_std = np.std(scores)
            max_std = 2.0  # Maximum possible std for scores from -1 to 1
            
            # Base confidence from agreement
            agreement_confidence = max(0, (max_std - score_std) / max_std)
            
            # Boost confidence for stronger signals
            signal_strength = abs(weighted_score)
            strength_confidence = signal_strength
            
            # Combined confidence
            confidence = (agreement_confidence * 0.6 + strength_confidence * 0.4) * 100
            
            return max(0, min(100, confidence))
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0

# Cell 28: Price targets and risk-reward calculation
    def _calculate_price_targets(self, state: Dict[str, Any], recommendation: str,
                               technical_data: Dict[str, Any], risk_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate price targets and stop-loss levels"""
        try:
            current_price = state.get('current_price', 100.0)
            support_resistance = technical_data.get('support_resistance', {})
            volatility = risk_data.get('volatility', 0.2)
            
            # Base target multipliers based on recommendation
            target_multipliers = {
                'STRONG BUY': 1.15,
                'BUY': 1.08,
                'HOLD': 1.02,
                'SELL': 0.95,
                'STRONG SELL': 0.85
            }
            
            base_target = current_price * target_multipliers.get(recommendation, 1.0)
            
            # Adjust based on technical levels
            resistance = support_resistance.get('nearest_resistance', current_price * 1.05)
            support = support_resistance.get('nearest_support', current_price * 0.95)
            
            # For buy recommendations, target near resistance
            if recommendation in ['STRONG BUY', 'BUY']:
                if resistance > current_price:
                    price_target = min(base_target, resistance * 0.98)  # Slightly below resistance
                else:
                    price_target = base_target
                
                # Stop loss below support
                stop_loss = max(support * 0.98, current_price * 0.92)
            
            # For sell recommendations, target near support
            elif recommendation in ['SELL', 'STRONG SELL']:
                if support < current_price:
                    price_target = max(base_target, support * 1.02)  # Slightly above support
                else:
                    price_target = base_target
                
                # Stop loss above resistance
                stop_loss = min(resistance * 1.02, current_price * 1.08)
            
            # For hold, narrow targets
            else:
                price_target = current_price * 1.02
                stop_loss = current_price * 0.95
            
            # Calculate percentages
            upside_percent = ((price_target - current_price) / current_price) * 100
            downside_percent = ((current_price - stop_loss) / current_price) * 100
            
            # Risk-reward ratio
            risk_reward_ratio = abs(upside_percent / downside_percent) if downside_percent != 0 else 0
            
            return {
                'target': round(price_target, 2),
                'stop_loss': round(stop_loss, 2),
                'upside_percent': round(upside_percent, 2),
                'downside_percent': round(downside_percent, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2)
            }
        except Exception as e:
            logger.error(f"Error calculating price targets: {e}")
            return {
                'target': state.get('current_price', 100) * 1.05,
                'stop_loss': state.get('current_price', 100) * 0.95,
                'upside_percent': 5.0,
                'downside_percent': 5.0,
                'risk_reward_ratio': 1.0
            }

# Cell 29: Rationale generation and time horizon
    def _generate_rationale(self, recommendation: str, sentiment_data: Dict[str, Any],
                          technical_data: Dict[str, Any], risk_data: Dict[str, Any],
                          fundamental_data: Dict[str, Any], confidence: float) -> str:
        """Generate detailed rationale for the recommendation"""
        try:
            rationale_parts = []
            
            # Opening statement
            rationale_parts.append(f"Recommendation: {recommendation} (Confidence: {confidence:.0f}%)")
            
            # Sentiment analysis
            overall_sentiment = sentiment_data.get('overall_sentiment', 0)
            if abs(overall_sentiment) > 0.2:
                sentiment_desc = "positive" if overall_sentiment > 0 else "negative"
                rationale_parts.append(f"Market sentiment is {sentiment_desc} ({overall_sentiment:.2f}), indicating {sentiment_desc} investor perception.")
            
            # Technical analysis
            trend = technical_data.get('trend_direction', 'Neutral')
            if trend != 'Neutral':
                rationale_parts.append(f"Technical analysis shows {trend.lower()} trend with supporting indicators.")
            
            patterns = technical_data.get('chart_patterns', [])
            if patterns:
                rationale_parts.append(f"Identified chart patterns: {', '.join(patterns)}.")
            
            # Risk assessment
            risk_score = risk_data.get('risk_score', 0.5)
            risk_level = "high" if risk_score > 0.6 else "moderate" if risk_score > 0.3 else "low"
            rationale_parts.append(f"Risk assessment indicates {risk_level} risk profile (score: {risk_score:.2f}).")
            
            # Fundamental factors
            ratios = fundamental_data.get('financial_data', {}).get('financial_ratios', {})
            if ratios:
                pe_ratio = ratios.get('pe_ratio', 0)
                if pe_ratio > 0:
                    rationale_parts.append(f"P/E ratio of {pe_ratio:.1f} suggests {'expensive' if pe_ratio > 25 else 'reasonable' if pe_ratio > 15 else 'attractive'} valuation.")
            
            # Risk-reward consideration
            if recommendation in ['STRONG BUY', 'BUY']:
                rationale_parts.append("The positive factors outweigh the risks, presenting a favorable risk-adjusted opportunity.")
            elif recommendation in ['SELL', 'STRONG SELL']:
                rationale_parts.append("Risk factors and negative indicators suggest limited upside potential and increased downside risk.")
            else:
                rationale_parts.append("Mixed signals suggest a cautious approach with careful monitoring of key indicators.")
            
            return " ".join(rationale_parts)
        except Exception as e:
            logger.error(f"Error generating rationale: {e}")
            return f"Recommendation: {recommendation} based on comprehensive analysis of available data."
    
    def _determine_time_horizon(self, technical_data: Dict[str, Any], recommendation: str) -> str:
        """Determine recommended time horizon for the investment"""
        try:
            trend = technical_data.get('trend_direction', 'Neutral')
            patterns = technical_data.get('chart_patterns', [])
            
            # Strong trends suggest longer holding periods
            if recommendation in ['STRONG BUY', 'STRONG SELL']:
                if trend in ['Upward', 'Downward']:
                    return 'Medium to Long term (3-12 months)'
                else:
                    return 'Short to Medium term (1-6 months)'
            elif recommendation in ['BUY', 'SELL']:
                return 'Short to Medium term (1-6 months)'
            else:
                return 'Short term (1-3 months)'
        except:
            return 'Medium term (3-6 months)'
    
    def _default_decision(self) -> Dict[str, Any]:
        """Return default decision in case of errors"""
        return {
            'recommendation': 'HOLD',
            'confidence_score': 50.0,
            'weighted_score': 0.0,
            'component_scores': {},
            'price_target': 0.0,
            'stop_loss': 0.0,
            'upside_potential': 0.0,
            'downside_risk': 0.0,
            'risk_reward_ratio': 1.0,
            'rationale': 'Unable to complete analysis due to insufficient data.',
            'time_horizon': 'Medium term'
        }


# ### Cell 26-29: Implement Trade Execution Agent (Kite Connect)



# Cell 26: Trade Execution Agent — Kite Connect (NSE, paper/live)

class TradeExecutionAgent:
    """Agent that executes trades via Kite Connect after a high-confidence recommendation."""

    def __init__(self, api_config: APIConfig, trading_config: TradingConfig):
        self.api_config = api_config
        self.trading_config = trading_config
        self.kite = None

        if not trading_config.paper_trading:
            if KiteConnect is None:
                raise ImportError("kiteconnect not installed. Run: pip install kiteconnect")
            if not api_config.kite_api_key or not api_config.kite_access_token:
                raise ValueError("kite_api_key and kite_access_token must be set for live trading")
            self.kite = KiteConnect(api_key=api_config.kite_api_key)
            self.kite.set_access_token(api_config.kite_access_token)
            logger.info("Kite Connect initialised for LIVE trading on %s", trading_config.exchange)
        else:
            logger.info("TradeExecutionAgent initialised in PAPER TRADING mode")

    # ── helpers ──────────────────────────────────────────────────────────────

    def calculate_quantity(self, current_price: float) -> int:
        """Derive order quantity from max_order_value; fall back to default_quantity."""
        if self.trading_config.max_order_value > 0 and current_price > 0:
            return max(1, int(self.trading_config.max_order_value / current_price))
        return self.trading_config.default_quantity

    def _clean_symbol(self, symbol: str) -> str:
        """Strip exchange suffix (.NS, .BSE) — Kite Connect uses bare symbols."""
        return symbol.split(".")[0]

    # ── public entry point ────────────────────────────────────────────────────

    def execute_trade(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read recommendation + confidence from state and place an order if eligible.

        Skips trade when:
          - confidence_score < confidence_threshold (default 80)
          - recommendation == 'HOLD'
        """
        symbol = state["symbol"]
        kite_symbol = self._clean_symbol(symbol)   # e.g. RELIANCE.NS -> RELIANCE
        recommendation = state.get("recommendation", "HOLD")
        confidence_score = state.get("confidence_score", 0.0)
        current_price = state.get("current_price", 0.0)
        stop_loss = state.get("stop_loss", 0.0)
        price_target = state.get("price_target", 0.0)

        if confidence_score < self.trading_config.confidence_threshold:
            logger.info(
                "Skipping trade for %s: confidence %.1f < threshold %.1f",
                symbol, confidence_score, self.trading_config.confidence_threshold
            )
            return {
                "executed": False,
                "reason": (
                    f"Confidence {confidence_score:.1f} below threshold "
                    f"{self.trading_config.confidence_threshold}"
                ),
                "recommendation": recommendation,
                "confidence_score": confidence_score,
            }

        if recommendation == "HOLD":
            logger.info("No trade for %s: recommendation is HOLD", symbol)
            return {
                "executed": False,
                "reason": "Recommendation is HOLD — no trade placed",
                "recommendation": recommendation,
                "confidence_score": confidence_score,
            }

        quantity = self.calculate_quantity(current_price)
        transaction_type = "BUY" if recommendation == "BUY" else "SELL"

        if self.trading_config.paper_trading:
            return self._place_paper_trade(
                kite_symbol, transaction_type, quantity,
                current_price, stop_loss, price_target, confidence_score
            )
        return self._place_live_trade(
            kite_symbol, transaction_type, quantity,
            confidence_score, stop_loss, price_target
        )

    # ── paper trading ─────────────────────────────────────────────────────────

    def _place_paper_trade(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        current_price: float,
        stop_loss: float,
        price_target: float,
        confidence_score: float,
    ) -> Dict[str, Any]:
        order_value = quantity * current_price
        order_id = f"PAPER-{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        logger.info(
            "[PAPER TRADE] %s %d x %s @ INR %.2f  "
            "(value: INR %.2f | SL: INR %.2f | target: INR %.2f) [%s]",
            transaction_type, quantity, symbol, current_price,
            order_value, stop_loss, price_target, order_id
        )

        return {
            "executed": True,
            "paper_trade": True,
            "order_id": order_id,
            "symbol": symbol,
            "exchange": self.trading_config.exchange,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": self.trading_config.order_type,
            "product": self.trading_config.product,
            "price": current_price,
            "order_value": order_value,
            "stop_loss": stop_loss,
            "price_target": price_target,
            "confidence_score": confidence_score,
            "timestamp": datetime.now().isoformat(),
            "status": "PAPER_COMPLETE",
        }

    # ── live trading ──────────────────────────────────────────────────────────

    def _place_live_trade(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        confidence_score: float,
        stop_loss: float,
        price_target: float,
    ) -> Dict[str, Any]:
        try:
            kite_txn = (
                self.kite.TRANSACTION_TYPE_BUY
                if transaction_type == "BUY"
                else self.kite.TRANSACTION_TYPE_SELL
            )
            kite_order_type = (
                self.kite.ORDER_TYPE_MARKET
                if self.trading_config.order_type == "MARKET"
                else self.kite.ORDER_TYPE_LIMIT
            )
            kite_product = (
                self.kite.PRODUCT_CNC
                if self.trading_config.product == "CNC"
                else self.kite.PRODUCT_MIS
            )

            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                tradingsymbol=symbol,
                exchange=self.trading_config.exchange,
                transaction_type=kite_txn,
                quantity=quantity,
                order_type=kite_order_type,
                product=kite_product,
            )

            logger.info(
                "[LIVE ORDER] %s %d x %s on %s [order_id: %s]",
                transaction_type, quantity, symbol, self.trading_config.exchange, order_id
            )

            return {
                "executed": True,
                "paper_trade": False,
                "order_id": str(order_id),
                "symbol": symbol,
                "exchange": self.trading_config.exchange,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": self.trading_config.order_type,
                "product": self.trading_config.product,
                "stop_loss": stop_loss,
                "price_target": price_target,
                "confidence_score": confidence_score,
                "timestamp": datetime.now().isoformat(),
                "status": "ORDER_PLACED",
            }

        except Exception as exc:
            error_msg = f"Live trade failed for {symbol}: {exc}"
            logger.error(error_msg)
            return {
                "executed": False,
                "paper_trade": False,
                "error": error_msg,
                "symbol": symbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "timestamp": datetime.now().isoformat(),
                "status": "FAILED",
            }


# ── LangGraph node factory ────────────────────────────────────────────────────

def create_trade_execution_node(agent: TradeExecutionAgent):
    """Create trade execution node for LangGraph."""
    def trade_execution_node(state: StockAnalysisState) -> StockAnalysisState:
        try:
            start_time = datetime.now()

            trade_result = agent.execute_trade(state)

            state["trade_executed"] = trade_result.get("executed", False)
            state["trade_details"] = trade_result
            state["order_id"] = trade_result.get("order_id")

            execution_time = (datetime.now() - start_time).total_seconds()
            state["execution_time"]["trade_execution"] = execution_time

            # Recalculate total (exclude existing 'total' key to avoid double-counting)
            state["execution_time"]["total"] = sum(
                v for k, v in state["execution_time"].items() if k != "total"
            )

            status = "executed" if trade_result.get("executed") else "skipped"
            logger.info(
                "Trade execution %s for %s in %.2fs",
                status, state["symbol"], execution_time
            )
            return state

        except Exception as exc:
            error_msg = f"Trade execution error: {exc}"
            state["errors"].append(error_msg)
            state["trade_executed"] = False
            state["trade_details"] = {"error": error_msg}
            state["order_id"] = None
            logger.error(error_msg)
            return state

    return trade_execution_node


# ### Cell 30-34: Set up LangGraph Workflow



# Cell 30: LangGraph workflow setup
from langgraph.graph import StateGraph, END
from typing import Dict, Any

def setup_stock_analysis_workflow(
    api_config: APIConfig,
    analysis_config: AnalysisConfig,
    trading_config: Optional[TradingConfig] = None,
) -> StateGraph:
    """Set up the LangGraph workflow for stock analysis"""
    if trading_config is None:
        trading_config = TradingConfig()

    # Initialise agents
    data_agent      = DataAnalysisAgent(api_config)
    sentiment_agent = SentimentAnalysisAgent(api_config)
    technical_agent = TechnicalAnalysisAgent(analysis_config)
    risk_agent      = RiskAssessmentAgent(analysis_config)
    decision_agent  = DecisionMakingAgent(analysis_config)
    trade_agent     = TradeExecutionAgent(api_config, trading_config)

    # Build graph
    workflow = StateGraph(StockAnalysisState)

    workflow.add_node("data_analysis",     create_data_analysis_node(data_agent))
    workflow.add_node("sentiment_analysis",create_sentiment_analysis_node(sentiment_agent))
    workflow.add_node("technical_analysis",create_technical_analysis_node(technical_agent))
    workflow.add_node("risk_assessment",   create_risk_assessment_node(risk_agent))
    workflow.add_node("decision_making",   create_decision_making_node(decision_agent))
    workflow.add_node("trade_execution",   create_trade_execution_node(trade_agent))

    workflow.set_entry_point("data_analysis")

    workflow.add_edge("data_analysis",     "sentiment_analysis")
    workflow.add_edge("data_analysis",     "technical_analysis")
    workflow.add_edge("sentiment_analysis","risk_assessment")
    workflow.add_edge("technical_analysis","risk_assessment")
    workflow.add_edge("risk_assessment",   "decision_making")
    workflow.add_edge("decision_making",   "trade_execution")
    workflow.add_edge("trade_execution",   END)

    return workflow.compile()

# Cell 31: Node creation functions
def create_data_analysis_node(agent: DataAnalysisAgent):
    def data_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
        try:
            start_time = datetime.now()
            symbol = state["symbol"]

            stock_data = agent.fetch_stock_data(symbol)
            if not stock_data:
                state["errors"].append("Failed to fetch stock data")
                return state

            state.update({
                "company_name":        stock_data.get("company_name", symbol),
                "current_price":       stock_data.get("current_price", 0),
                "price_change":        stock_data.get("price_change", 0),
                "price_change_percent":stock_data.get("price_change_percent", 0),
                "volume":              stock_data.get("volume", 0),
                "market_cap":          stock_data.get("market_cap", 0),
                "historical_data":     stock_data.get("historical_data", pd.DataFrame()),
            })

            state["financial_data"] = agent.get_financial_data(symbol)
            state.setdefault("market_context", {}).update(agent.get_market_context(symbol))
            state.setdefault("economic_data",  {}).update(agent.get_economic_indicators())

            execution_time = (datetime.now() - start_time).total_seconds()
            state["execution_time"]["data_analysis"] = execution_time
            logger.info(f"Data analysis completed for {symbol} in {execution_time:.2f}s")
            return state

        except Exception as e:
            error_msg = f"Data analysis error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state

    return data_analysis_node


def create_sentiment_analysis_node(agent: SentimentAnalysisAgent):
    def sentiment_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
        try:
            start_time = datetime.now()
            symbol = state["symbol"]

            news_analysis   = agent.analyze_news_sentiment(symbol)
            social_analysis = agent.analyze_social_sentiment(symbol)
            analyst_analysis= agent.get_analyst_sentiment(symbol)

            state["news_sentiment"]    = news_analysis.get("sentiment_score", 0.0)
            state["social_sentiment"]  = social_analysis.get("sentiment_score", 0.0)
            state["analyst_consensus"] = analyst_analysis.get("sentiment_score", 0.0)

            overall = agent.calculate_overall_sentiment(
                state["news_sentiment"],
                state["social_sentiment"],
                state["analyst_consensus"],
            )
            state["overall_sentiment"] = overall.get("overall_sentiment", 0.0)
            state["sentiment_details"] = {
                "news_analysis":    news_analysis,
                "social_analysis":  social_analysis,
                "analyst_analysis": analyst_analysis,
                "overall_analysis": overall,
            }

            execution_time = (datetime.now() - start_time).total_seconds()
            state["execution_time"]["sentiment_analysis"] = execution_time
            logger.info(f"Sentiment analysis completed for {symbol} in {execution_time:.2f}s")
            return state

        except Exception as e:
            error_msg = f"Sentiment analysis error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state

    return sentiment_analysis_node


def create_technical_analysis_node(agent: TechnicalAnalysisAgent):
    def technical_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
        try:
            start_time = datetime.now()
            historical_data = state.get("historical_data", pd.DataFrame())

            if historical_data.empty:
                state["warnings"].append("No historical data available for technical analysis")
                return state

            indicators       = agent.calculate_all_indicators(historical_data)
            support_resistance = agent.identify_support_resistance(historical_data)
            trend_analysis   = agent.analyze_trend_and_patterns(historical_data)

            state["technical_indicators"] = indicators
            state["support_resistance"]   = support_resistance
            state["trend_direction"]      = trend_analysis.get("overall_trend", "Neutral")
            state["chart_patterns"]       = trend_analysis.get("patterns", [])
            state.setdefault("technical_analysis_details", {}).update({
                "indicators": indicators,
                "support_resistance": support_resistance,
                "trend_analysis": trend_analysis,
            })

            execution_time = (datetime.now() - start_time).total_seconds()
            state["execution_time"]["technical_analysis"] = execution_time
            logger.info(f"Technical analysis completed in {execution_time:.2f}s")
            return state

        except Exception as e:
            error_msg = f"Technical analysis error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state

    return technical_analysis_node


def create_risk_assessment_node(agent: RiskAssessmentAgent):
    def risk_assessment_node(state: StockAnalysisState) -> StockAnalysisState:
        try:
            start_time = datetime.now()
            symbol = state["symbol"]
            historical_data = state.get("historical_data", pd.DataFrame())

            if historical_data.empty:
                state["warnings"].append("No historical data available for risk assessment")
                return state

            market_data = {
                "market_cap":   state.get("market_cap", 0),
                "financial_data": state.get("financial_data", {}),
            }
            risk_metrics = agent.calculate_comprehensive_risk(historical_data, symbol, market_data)

            state["risk_score"] = risk_metrics.get("overall_risk_score", {}).get("overall_risk_score", 0.5)
            state["volatility"] = risk_metrics.get("annual_volatility", 0.2)
            state["beta"]       = risk_metrics.get("beta", 1.0)
            state["var_score"]  = risk_metrics.get("var_historical_95", -0.02)
            state["risk_metrics"] = risk_metrics

            execution_time = (datetime.now() - start_time).total_seconds()
            state["execution_time"]["risk_assessment"] = execution_time
            logger.info(f"Risk assessment completed for {symbol} in {execution_time:.2f}s")
            return state

        except Exception as e:
            error_msg = f"Risk assessment error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state

    return risk_assessment_node


def create_decision_making_node(agent: DecisionMakingAgent):
    def decision_making_node(state: StockAnalysisState) -> StockAnalysisState:
        try:
            start_time = datetime.now()

            decision = agent.make_investment_decision(state)

            state.update({
                "recommendation":  decision.get("recommendation", "HOLD"),
                "confidence_score":decision.get("confidence_score", 50.0),
                "price_target":    decision.get("price_target", 0.0),
                "stop_loss":       decision.get("stop_loss", 0.0),
                "rationale":       decision.get("rationale", "Analysis completed"),
            })
            state.setdefault("decision_details", {}).update(decision)

            execution_time = (datetime.now() - start_time).total_seconds()
            state["execution_time"]["decision_making"] = execution_time
            logger.info(f"Decision making completed in {execution_time:.2f}s")
            return state

        except Exception as e:
            error_msg = f"Decision making error: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            return state

    return decision_making_node


class StockAnalysisWorkflow:
    """Main workflow orchestrator for stock analysis"""

    def __init__(
        self,
        api_config: APIConfig,
        analysis_config: AnalysisConfig,
        trading_config: Optional[TradingConfig] = None,
    ):
        self.api_config     = api_config
        self.analysis_config = analysis_config
        self.trading_config  = trading_config or TradingConfig()
        self.workflow = setup_stock_analysis_workflow(
            api_config, analysis_config, self.trading_config
        )

    def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """Analyse a single stock and optionally execute a trade."""
        try:
            initial_state = StockAnalysisState(
                symbol=symbol.upper(),
                company_name="",
                analysis_date=datetime.now(),
                current_price=0.0,
                price_change=0.0,
                price_change_percent=0.0,
                volume=0,
                market_cap=0.0,
                historical_data=pd.DataFrame(),
                financial_data={},
                news_sentiment=0.0,
                social_sentiment=0.0,
                analyst_consensus=0.0,
                overall_sentiment=0.0,
                sentiment_details={},
                technical_indicators={},
                support_resistance={},
                trend_direction="Neutral",
                chart_patterns=[],
                risk_score=0.5,
                volatility=0.2,
                beta=1.0,
                var_score=-0.02,
                risk_metrics={},
                recommendation="HOLD",
                confidence_score=50.0,
                price_target=0.0,
                stop_loss=0.0,
                rationale="",
                trade_executed=False,
                trade_details={},
                order_id=None,
                errors=[],
                warnings=[],
                execution_time={},
            )

            logger.info(f"Starting analysis for {symbol}")
            result = self.workflow.invoke(initial_state)

            total_time = result.get("execution_time", {}).get("total", 0)
            logger.info(f"Analysis completed for {symbol} in {total_time:.2f}s")

            if result.get("errors"):
                logger.warning(f"Errors during analysis: {result['errors']}")
            if result.get("warnings"):
                logger.info(f"Warnings during analysis: {result['warnings']}")

            return result

        except Exception as e:
            logger.error(f"Workflow execution error for {symbol}: {e}")
            return {
                "symbol":           symbol,
                "recommendation":   "HOLD",
                "confidence_score": 0.0,
                "rationale":        f"Analysis failed: {str(e)}",
                "trade_executed":   False,
                "trade_details":    {},
                "order_id":         None,
                "errors":           [str(e)],
            }

    def analyze_multiple_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyse multiple stocks sequentially."""
        results = {}
        for symbol in symbols:
            try:
                logger.info(f"Analysing {symbol}...")
                results[symbol] = self.analyze_stock(symbol)
            except Exception as e:
                logger.error(f"Failed to analyse {symbol}: {e}")
                results[symbol] = {
                    "symbol":           symbol,
                    "recommendation":   "HOLD",
                    "confidence_score": 0.0,
                    "rationale":        f"Analysis failed: {str(e)}",
                    "trade_executed":   False,
                    "trade_details":    {},
                    "order_id":         None,
                    "errors":           [str(e)],
                }
        return results

    def get_workflow_status(self) -> Dict[str, Any]:
        """Return workflow configuration summary."""
        return {
            "api_config_set":       bool(self.api_config.alpha_vantage_key or self.api_config.news_api_key),
            "workflow_ready":       self.workflow is not None,
            "trading_mode":         "paper" if self.trading_config.paper_trading else "live",
            "confidence_threshold": self.trading_config.confidence_threshold,
            "exchange":             self.trading_config.exchange,
            "supported_analysis": [
                "data_analysis",
                "sentiment_analysis",
                "technical_analysis",
                "risk_assessment",
                "decision_making",
                "trade_execution",
            ],
        }


# Cell 34: Utility functions
def create_default_configs() -> tuple[APIConfig, AnalysisConfig, TradingConfig]:
    """Create default configurations from environment variables."""
    api_config = APIConfig(
        alpha_vantage_key    =os.getenv("ALPHA_VANTAGE_API_KEY", ""),
        news_api_key         =os.getenv("NEWS_API_KEY", ""),
        fred_api_key         =os.getenv("FRED_API_KEY", ""),
        twitter_bearer_token =os.getenv("TWITTER_BEARER_TOKEN", ""),
        reddit_client_id     =os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret =os.getenv("REDDIT_CLIENT_SECRET", ""),
        openai_api_key       =os.getenv("OPENAI_API_KEY", ""),
        kite_api_key         =os.getenv("KITE_API_KEY", ""),
        kite_access_token    =os.getenv("KITE_ACCESS_TOKEN", ""),
    )

    analysis_config = AnalysisConfig(
        lookback_days=252,
        short_ma_period=20,
        long_ma_period=50,
        rsi_period=14,
        bollinger_period=20,
        confidence_threshold=0.6,
        risk_free_rate=0.02,
    )

    trading_config = TradingConfig(
        paper_trading=True,
        confidence_threshold=80.0,
        exchange="NSE",
        default_quantity=1,
        max_order_value=50000.0,
        order_type="MARKET",
        product="CNC",
    )

    return api_config, analysis_config, trading_config


def validate_configuration(api_config: APIConfig) -> Dict[str, bool]:
    """Validate which API keys are configured."""
    return {
        "yfinance_available":     True,
        "alpha_vantage_configured":bool(api_config.alpha_vantage_key),
        "news_api_configured":    bool(api_config.news_api_key),
        "fred_api_configured":    bool(api_config.fred_api_key),
        "twitter_api_configured": bool(api_config.twitter_bearer_token),
        "reddit_api_configured":  bool(api_config.reddit_client_id and api_config.reddit_client_secret),
        "openai_configured":      bool(api_config.openai_api_key),
        "kite_configured":        bool(api_config.kite_api_key and api_config.kite_access_token),
    }


def setup_logging_config(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('stock_analysis.log')],
    )


async def async_analyze_stocks(workflow: StockAnalysisWorkflow, symbols: List[str]) -> Dict[str, Any]:
    """Async wrapper (placeholder — runs synchronously)."""
    return workflow.analyze_multiple_stocks(symbols)


def create_workflow_visualization() -> str:
    return """
    Stock Analysis Workflow:

    [Input: Stock Symbol]
            |
    [Data Analysis Agent] ──────┐
            |                   |
            v                   v
    [Sentiment Analysis]   [Technical Analysis]
            |                   |
            └─────┬─────────────┘
                  v
        [Risk Assessment Agent]
                  |
                  v
        [Decision Making Agent]
                  |
                  v
    [Trade Execution Agent]  <- Kite Connect / NSE
     confidence >= 80 → BUY or SELL order placed
     confidence < 80 or HOLD → skipped
                  |
                  v
    [Output: recommendation + order_id + trade_details]
    """


def demo_workflow_setup():
    print("Setting up Stock Analysis Workflow...")

    api_config, analysis_config, trading_config = create_default_configs()

    validation = validate_configuration(api_config)
    print("Configuration Validation:")
    for service, status in validation.items():
        print(f"  {service}: {'OK' if status else 'NOT SET'}")

    try:
        workflow = StockAnalysisWorkflow(api_config, analysis_config, trading_config)
        print("Workflow setup successful")

        status = workflow.get_workflow_status()
        print(f"Workflow ready:        {status['workflow_ready']}")
        print(f"Trading mode:          {status['trading_mode']}")
        print(f"Confidence threshold:  {status['confidence_threshold']}")
        print(f"Supported analyses:    {', '.join(status['supported_analysis'])}")
        return workflow
    except Exception as e:
        print(f"Workflow setup failed: {e}")
        return None


print(create_workflow_visualization())



































