import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pickle
import io
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="⚽ Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for Windows-inspired styling
st.markdown("""
<style>
        .main {
            padding-top: 1rem;
        }
        .stMetric {
            background-color: #f1f1f1; /* Light gray background for a more Windows-like feel */
            border: 1px solid #dcdcdc; /* Soft border color */
            border-radius: 5px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        .metric-card {
            background: linear-gradient(135deg, #0078d4 0%, #2a8bcf 100%); /* Windows blue gradient */
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Adding shadow */
        }
        .stSelectbox > div > div {
            background-color: #ffffff; /* Cleaner white for the dropdown */
            border: 1px solid #cccccc; /* Lighter border for dropdown */
            border-radius: 5px;
        }
        .big-font {
            font-size: 24px !important;
            font-weight: bold;
            color: #1e1e1e; /* Darker text color for Windows-style */
        }
        .medium-font {
            font-size: 18px !important;
            font-weight: 600;
            color: #3c3c3c; /* Medium gray text color */
        }
        .highlight {
            background-color: #0078d4; /* Windows primary blue */
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 5px;
            font-weight: bold;
        }
        .win-card {
            background: linear-gradient(135deg, #0078d4 0%, #2a8bcf 100%); /* Windows blue */
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }
        .draw-card {
            background: linear-gradient(135deg, #f3c200 0%, #ff9800 100%); /* Windows amber yellow */
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }
        .loss-card {
            background: linear-gradient(135deg, #e81123 0%, #f37059 100%); /* Windows red */
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the football dataset"""
    try:
        df = pd.read_csv('data.csv')
        
        # Basic data cleaning
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        
        # Clean string columns
        string_columns = ['home_team', 'away_team', 'tournament', 'city', 'country']
        for col in string_columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', 'Unknown')
            df[col] = df[col].replace('', 'Unknown')
        
        # Create derived features
        df['total_goals'] = df['home_score'] + df['away_score']
        df['goal_difference'] = df['home_score'] - df['away_score']
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['decade'] = (df['year'] // 10) * 10
        
        # Create match outcome
        df['outcome'] = df.apply(lambda row: 
            'Home Win' if row['home_score'] > row['away_score'] 
            else 'Away Win' if row['home_score'] < row['away_score'] 
            else 'Draw', axis=1)
        
        # Create high/low scoring match indicator
        df['high_scoring'] = df['total_goals'] > df['total_goals'].median()
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['date', 'home_score', 'away_score'])
        df = df[df['home_team'] != 'Unknown']
        df = df[df['away_team'] != 'Unknown']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_sidebar_filters(df):
    """Create sidebar filters with ALL option by default"""
    st.sidebar.header("⚽ Filters")
    
    try:
        # Date range filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Team filters
        all_teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
        teams = ['ALL'] + [team for team in all_teams if team != 'Unknown']
        selected_team = st.sidebar.selectbox("Select Team", teams, index=0)
        
        # Tournament filter
        tournaments = ['ALL'] + sorted([t for t in df['tournament'].unique() if t != 'Unknown'])
        selected_tournament = st.sidebar.selectbox("Select Tournament", tournaments, index=0)
        
        # Country filter
        countries = ['ALL'] + sorted([c for c in df['country'].unique() if c != 'Unknown'])
        selected_country = st.sidebar.selectbox("Select Country", countries, index=0)
        
        # Match outcome filter
        outcomes = ['ALL', 'Home Win', 'Draw', 'Away Win']
        selected_outcome = st.sidebar.selectbox("Select Match Outcome", outcomes, index=0)
        
        # Goals range filter
        min_goals, max_goals = int(df['total_goals'].min()), int(df['total_goals'].max())
        goals_range = st.sidebar.slider(
            "Total Goals Range",
            min_value=min_goals,
            max_value=max_goals,
            value=(min_goals, max_goals)
        )
        
        # Neutral venue filter
        neutral_filter = st.sidebar.selectbox("Venue Type", ['ALL', 'Neutral', 'Home/Away'])
        
    except Exception as e:
        st.sidebar.error(f"Error creating filters: {e}")
        # Fallback values
        date_range = (df['date'].min().date(), df['date'].max().date())
        selected_team = 'ALL'
        selected_tournament = 'ALL'
        selected_country = 'ALL'
        selected_outcome = 'ALL'
        goals_range = (0, 20)
        neutral_filter = 'ALL'
    
    return {
        'date_range': date_range,
        'team': selected_team,
        'tournament': selected_tournament,
        'country': selected_country,
        'outcome': selected_outcome,
        'goals_range': goals_range,
        'neutral': neutral_filter
    }

def apply_filters(df, filters):
    """Apply selected filters to the dataframe"""
    filtered_df = df.copy()
    
    # Date filter
    if len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Team filter
    if filters['team'] != 'ALL':
        filtered_df = filtered_df[
            (filtered_df['home_team'] == filters['team']) | 
            (filtered_df['away_team'] == filters['team'])
        ]
    
    # Tournament filter
    if filters['tournament'] != 'ALL':
        filtered_df = filtered_df[filtered_df['tournament'] == filters['tournament']]
    
    # Country filter
    if filters['country'] != 'ALL':
        filtered_df = filtered_df[filtered_df['country'] == filters['country']]
    
    # Outcome filter
    if filters['outcome'] != 'ALL':
        filtered_df = filtered_df[filtered_df['outcome'] == filters['outcome']]
    
    # Goals range filter
    filtered_df = filtered_df[
        (filtered_df['total_goals'] >= filters['goals_range'][0]) & 
        (filtered_df['total_goals'] <= filters['goals_range'][1])
    ]
    
    # Neutral venue filter
    if filters['neutral'] == 'Neutral':
        filtered_df = filtered_df[filtered_df['neutral'] == True]
    elif filters['neutral'] == 'Home/Away':
        filtered_df = filtered_df[filtered_df['neutral'] == False]
    
    return filtered_df

def data_overview_section(df):
    """Data Overview Section"""
    st.header("⚽ Football Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Matches</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Unique Teams</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(set(df['home_team'].unique()) | set(df['away_team'].unique()))), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Goals/Match</h3>
            <h2>{:.1f}</h2>
        </div>
        """.format(df['total_goals'].mean()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Years Covered</h3>
            <h2>{} - {}</h2>
        </div>
        """.format(df['year'].min(), df['year'].max()), unsafe_allow_html=True)
    
    # Match outcomes distribution
    st.subheader("📊 Match Outcomes Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    outcome_counts = df['outcome'].value_counts()
    total_matches = len(df)
    
    with col1:
        home_wins = outcome_counts.get('Home Win', 0)
        home_pct = (home_wins / total_matches) * 100
        st.markdown(f"""
        <div class="win-card">
            <h3>Home Wins</h3>
            <h2>{home_wins:,}</h2>
            <p>{home_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        draws = outcome_counts.get('Draw', 0)
        draw_pct = (draws / total_matches) * 100
        st.markdown(f"""
        <div class="draw-card">
            <h3>Draws</h3>
            <h2>{draws:,}</h2>
            <p>{draw_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        away_wins = outcome_counts.get('Away Win', 0)
        away_pct = (away_wins / total_matches) * 100
        st.markdown(f"""
        <div class="loss-card">
            <h3>Away Wins</h3>
            <h2>{away_wins:,}</h2>
            <p>{away_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset summary
    st.subheader("📋 Dataset Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Historical Statistics:**")
        stats_df = pd.DataFrame({
            'Metric': ['Total Goals Scored', 'Highest Scoring Match', 'Most Common Score', 
                      'Average Goals/Match', 'Home Advantage %'],
            'Value': [
                f"{df['total_goals'].sum():,}",
                f"{df['total_goals'].max()} goals",
                f"{df['home_score'].mode().iloc[0]}-{df['away_score'].mode().iloc[0]}",
                f"{df['total_goals'].mean():.2f}",
                f"{home_pct:.1f}%"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.write("**Tournament Information:**")
        tournament_info = df['tournament'].value_counts().head(10)
        st.dataframe(tournament_info.reset_index().rename(columns={'tournament': 'Tournament', 'count': 'Matches'}))
    
    # Sample data
    st.subheader("📄 Sample Data")
    sample_df = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 
                   'tournament', 'city', 'country', 'outcome']].head(10)
    st.dataframe(sample_df, use_container_width=True)

def eda_section(df):
    """Exploratory Data Analysis Section"""
    st.header("🔬 Exploratory Data Analysis")
    
    # Goals analysis
    st.subheader("⚽ Goals Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_goals_dist = px.histogram(
            df, x='total_goals', nbins=20,
            title="Distribution of Total Goals per Match",
            labels={'total_goals': 'Total Goals', 'count': 'Number of Matches'},
            color_discrete_sequence=['#28a745']
        )
        fig_goals_dist.update_layout(height=400)
        st.plotly_chart(fig_goals_dist, use_container_width=True)
    
    with col2:
        fig_outcome_pie = px.pie(
            df['outcome'].value_counts().reset_index(),
            values='count', names='outcome',
            title="Match Outcomes Distribution",
            color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
        )
        fig_outcome_pie.update_layout(height=400)
        st.plotly_chart(fig_outcome_pie, use_container_width=True)
    
    # Temporal analysis
    st.subheader("📅 Temporal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Goals over decades
        decade_goals = df.groupby('decade')['total_goals'].mean().reset_index()
        fig_decade = px.line(
            decade_goals, x='decade', y='total_goals',
            title="Average Goals per Match by Decade",
            labels={'decade': 'Decade', 'total_goals': 'Average Goals'},
            markers=True
        )
        fig_decade.update_layout(height=400)
        st.plotly_chart(fig_decade, use_container_width=True)
    
    with col2:
        # Matches by month
        month_matches = df.groupby('month').size().reset_index(name='matches')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_matches['month_name'] = month_matches['month'].map(lambda x: month_names[x-1])
        
        fig_monthly = px.bar(
            month_matches, x='month_name', y='matches',
            title="Matches by Month",
            labels={'month_name': 'Month', 'matches': 'Number of Matches'},
            color='matches',
            color_continuous_scale='viridis'
        )
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Team analysis
    st.subheader("🏆 Team Performance Analysis")
    
    # Calculate team statistics
    def calculate_team_stats(df):
        team_stats = []
        all_teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        
        for team in all_teams:
            if team == 'Unknown':
                continue
            
            home_matches = df[df['home_team'] == team]
            away_matches = df[df['away_team'] == team]
            
            total_matches = len(home_matches) + len(away_matches)
            if total_matches < 10:  # Only include teams with at least 10 matches
                continue
            
            # Goals scored and conceded
            goals_for = (home_matches['home_score'].sum() + away_matches['away_score'].sum())
            goals_against = (home_matches['away_score'].sum() + away_matches['home_score'].sum())
            
            # Wins, draws, losses
            home_wins = len(home_matches[home_matches['outcome'] == 'Home Win'])
            away_wins = len(away_matches[away_matches['outcome'] == 'Away Win'])
            home_draws = len(home_matches[home_matches['outcome'] == 'Draw'])
            away_draws = len(away_matches[away_matches['outcome'] == 'Draw'])
            
            wins = home_wins + away_wins
            draws = home_draws + away_draws
            losses = total_matches - wins - draws
            
            win_rate = (wins / total_matches) * 100 if total_matches > 0 else 0
            
            team_stats.append({
                'Team': team,
                'Matches': total_matches,
                'Wins': wins,
                'Draws': draws,
                'Losses': losses,
                'Goals For': goals_for,
                'Goals Against': goals_against,
                'Goal Difference': goals_for - goals_against,
                'Win Rate %': win_rate
            })
        
        return pd.DataFrame(team_stats)
    
    team_stats_df = calculate_team_stats(df)
    
    if not team_stats_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top teams by win rate
            top_teams = team_stats_df.nlargest(15, 'Win Rate %')
            fig_win_rate = px.bar(
                top_teams, x='Win Rate %', y='Team',
                title="Top 15 Teams by Win Rate",
                orientation='h',
                color='Win Rate %',
                color_continuous_scale='viridis'
            )
            fig_win_rate.update_layout(height=500)
            st.plotly_chart(fig_win_rate, use_container_width=True)
        
        with col2:
            # Goals for vs against
            fig_goals_scatter = px.scatter(
                team_stats_df, x='Goals For', y='Goals Against',
                size='Matches', hover_name='Team',
                title="Goals For vs Goals Against",
                color='Win Rate %',
                color_continuous_scale='RdYlGn'
            )
            fig_goals_scatter.update_layout(height=500)
            st.plotly_chart(fig_goals_scatter, use_container_width=True)
    
    # Tournament analysis
    st.subheader("🏆 Tournament Analysis")
    
    tournament_stats = df.groupby('tournament').agg({
        'total_goals': ['count', 'mean', 'sum'],
        'outcome': lambda x: (x == 'Home Win').mean() * 100
    }).round(2)
    
    tournament_stats.columns = ['Matches', 'Avg Goals', 'Total Goals', 'Home Win %']
    tournament_stats = tournament_stats.reset_index()
    tournament_stats = tournament_stats[tournament_stats['Matches'] >= 50]  # Filter tournaments with at least 50 matches
    tournament_stats = tournament_stats.sort_values('Matches', ascending=False).head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tournament_matches = px.bar(
            tournament_stats, x='tournament', y='Matches',
            title="Top 15 Tournaments by Number of Matches",
            color='Matches',
            color_continuous_scale='blues'
        )
        fig_tournament_matches.update_xaxes(tickangle=45)
        fig_tournament_matches.update_layout(height=500)
        st.plotly_chart(fig_tournament_matches, use_container_width=True)
    
    with col2:
        fig_tournament_goals = px.bar(
            tournament_stats, x='tournament', y='Avg Goals',
            title="Average Goals per Match by Tournament",
            color='Avg Goals',
            color_continuous_scale='reds'
        )
        fig_tournament_goals.update_xaxes(tickangle=45)
        fig_tournament_goals.update_layout(height=500)
        st.plotly_chart(fig_tournament_goals, use_container_width=True)

def visualization_section(df):
    """Advanced Data Visualization Section"""
    st.header("📊 Advanced Visualizations")
    
    # Goals heatmap by decade and tournament
    st.subheader("🔥 Goals Heatmap by Decade and Tournament")
    
    # Select top tournaments for better visualization
    top_tournaments = df['tournament'].value_counts().head(10).index
    df_filtered = df[df['tournament'].isin(top_tournaments)]
    
    heatmap_data = df_filtered.pivot_table(
        values='total_goals', 
        index='tournament', 
        columns='decade', 
        aggfunc='mean'
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Average Goals per Match: Tournament vs Decade",
        color_continuous_scale='Reds',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Home advantage analysis
    st.subheader("🏠 Home Advantage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Home advantage by tournament
        home_adv_tournament = df.groupby('tournament')['outcome'].apply(
            lambda x: (x == 'Home Win').mean() * 100
        ).reset_index()
        home_adv_tournament.columns = ['Tournament', 'Home Win %']
        home_adv_tournament = home_adv_tournament.sort_values('Home Win %', ascending=False).head(15)
        
        fig_home_adv = px.bar(
            home_adv_tournament, x='Home Win %', y='Tournament',
            title="Home Win Percentage by Tournament",
            orientation='h',
            color='Home Win %',
            color_continuous_scale='RdYlGn'
        )
        fig_home_adv.update_layout(height=500)
        st.plotly_chart(fig_home_adv, use_container_width=True)
    
    with col2:
        # Goals by venue type
        venue_goals = df.groupby('neutral').agg({
            'total_goals': 'mean',
            'home_score': 'mean',
            'away_score': 'mean'
        }).reset_index()
        venue_goals['venue_type'] = venue_goals['neutral'].map({True: 'Neutral', False: 'Home/Away'})
        
        fig_venue = px.bar(
            venue_goals, x='venue_type', y=['home_score', 'away_score'],
            title="Average Goals by Venue Type",
            barmode='group',
            color_discrete_sequence=['#28a745', '#dc3545']
        )
        fig_venue.update_layout(height=500)
        st.plotly_chart(fig_venue, use_container_width=True)
    
    # Timeline analysis
    st.subheader("📈 Historical Timeline")
    
    # Resample data by year for better visualization
    yearly_stats = df.groupby('year').agg({
        'total_goals': 'mean',
        'home_score': 'mean',
        'away_score': 'mean',
        'outcome': 'count'
    }).reset_index()
    yearly_stats.rename(columns={'outcome': 'matches'}, inplace=True)
    
    # Only show data from 1950 onwards for better visualization
    yearly_stats_modern = yearly_stats[yearly_stats['year'] >= 1950]
    
    fig_timeline = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Goals per Match Over Time', 'Number of Matches per Year'),
        shared_xaxes=True
    )
    
    fig_timeline.add_trace(
        go.Scatter(x=yearly_stats_modern['year'], y=yearly_stats_modern['total_goals'],
                  mode='lines+markers', name='Avg Goals', line=dict(color='#28a745')),
        row=1, col=1
    )
    
    fig_timeline.add_trace(
        go.Scatter(x=yearly_stats_modern['year'], y=yearly_stats_modern['matches'],
                  mode='lines+markers', name='Matches', line=dict(color='#dc3545')),
        row=2, col=1
    )
    
    fig_timeline.update_layout(height=600, title="Football Trends Over Time (1950-2025)")
    st.plotly_chart(fig_timeline, use_container_width=True)

@st.cache_data
def prepare_model_data(df):
    """Prepare data for machine learning models (optimized with caching)"""
    model_df = df.copy()
    
    # Smart sampling for large datasets (>15k rows)
    if len(model_df) > 15000:
        model_df = model_df.sample(n=15000, random_state=42)
        st.info(f"🔄 Using a sample of 15,000 rows for faster model training (original: {len(df):,} rows)")
    
    # Create features for prediction
    model_df['home_team_encoded'] = LabelEncoder().fit_transform(model_df['home_team'])
    model_df['away_team_encoded'] = LabelEncoder().fit_transform(model_df['away_team'])
    model_df['tournament_encoded'] = LabelEncoder().fit_transform(model_df['tournament'])
    model_df['country_encoded'] = LabelEncoder().fit_transform(model_df['country'])
    model_df['neutral_encoded'] = model_df['neutral'].astype(int)
    model_df['month_encoded'] = model_df['month']
    
    # Calculate team strength based on historical performance
    team_strength = {}
    all_teams = list(set(model_df['home_team'].unique()) | set(model_df['away_team'].unique()))
    
    for team in all_teams:
        team_matches = model_df[(model_df['home_team'] == team) | (model_df['away_team'] == team)]
        if len(team_matches) > 0:
            home_wins = len(team_matches[(team_matches['home_team'] == team) & (team_matches['outcome'] == 'Home Win')])
            away_wins = len(team_matches[(team_matches['away_team'] == team) & (team_matches['outcome'] == 'Away Win')])
            total_wins = home_wins + away_wins
            win_rate = total_wins / len(team_matches)
            team_strength[team] = win_rate
        else:
            team_strength[team] = 0.33  # Default strength
    
    model_df['home_team_strength'] = model_df['home_team'].map(team_strength)
    model_df['away_team_strength'] = model_df['away_team'].map(team_strength)
    model_df['strength_difference'] = model_df['home_team_strength'] - model_df['away_team_strength']
    
    # Features for prediction
    features = ['home_team_encoded', 'away_team_encoded', 'tournament_encoded', 
                'country_encoded', 'neutral_encoded', 'month_encoded',
                'home_team_strength', 'away_team_strength', 'strength_difference']
    
    X = model_df[features]
    
    # Create target variables
    y_outcome = model_df['outcome']  # Classification target
    y_total_goals = model_df['total_goals']  # Regression target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_outcome_encoded = label_encoder.fit_transform(y_outcome)
    
    return X_scaled, y_outcome_encoded, y_total_goals, features, {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'team_strength': team_strength
    }

def model_application_section(df):
    """Model Application Section with Top 3 Algorithms (Optimized)"""
    st.header("🤖 Machine Learning Model Application")
    
    # Performance optimization toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("⚡ Performance-optimized models for faster results")
    with col2:
        fast_mode = st.checkbox("Fast Mode", value=True, help="Reduces training time by using fewer estimators")
    
    # Model type selection
    prediction_type = st.radio(
        "Select Prediction Type:",
        ["Match Outcome (Win/Draw/Loss)", "Total Goals Prediction"],
        horizontal=True
    )
    
    # Prepare data
    with st.spinner("🔄 Preparing data..."):
        X, y_outcome, y_total_goals, features, encoders = prepare_model_data(df)
        
        if prediction_type == "Match Outcome (Win/Draw/Loss)":
            y = y_outcome
            model_type = "classification"
        else:
            y = y_total_goals
            model_type = "regression"
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define optimized models
    if fast_mode:
        if model_type == "classification":
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=50, max_depth=6, learning_rate=0.15, random_state=42
                ),
                'Logistic Regression': LogisticRegression(random_state=42, n_jobs=-1, max_iter=500)
            }
        else:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            models = {
                'Random Forest': RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=50, max_depth=6, learning_rate=0.15, random_state=42
                ),
                'Linear Regression': LinearRegression(n_jobs=-1)
            }
        cv_folds = 3
    else:
        if model_type == "classification":
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000)
            }
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(n_jobs=-1)
            }
        cv_folds = 5
    
    st.subheader(f"📈 {prediction_type} Model Comparison")
    
    model_results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train and evaluate models
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        progress_bar.progress((i) / len(models))
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        if model_type == "classification":
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Simplified cross-validation for speed
            if fast_mode:
                X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                model_temp = type(model)(**model.get_params())
                model_temp.fit(X_val_train, y_val_train)
                cv_score = model_temp.score(X_val_test, y_val_test)
                cv_scores_mean = cv_score
                cv_scores_std = 0.0
            else:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                cv_scores_mean = cv_scores.mean()
                cv_scores_std = cv_scores.std()
            
            model_results[name] = {
                'model': model,
                'Accuracy': accuracy,
                'F1 Score': f1,
                'CV Accuracy Mean': cv_scores_mean,
                'CV Accuracy Std': cv_scores_std,
                'predictions': y_pred
            }
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Simplified cross-validation for speed
            if fast_mode:
                X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                model_temp = type(model)(**model.get_params())
                model_temp.fit(X_val_train, y_val_train)
                cv_score = model_temp.score(X_val_test, y_val_test)
                cv_scores_mean = cv_score
                cv_scores_std = 0.0
            else:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                cv_scores_mean = cv_scores.mean()
                cv_scores_std = cv_scores.std()
            
            model_results[name] = {
                'model': model,
                'RMSE': rmse,
                'R² Score': r2,
                'MAE': mae,
                'CV R² Mean': cv_scores_mean,
                'CV R² Std': cv_scores_std,
                'predictions': y_pred
            }
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Model training completed!")
    
    # Performance summary
    st.subheader("⚡ Performance Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Trained", len(models))
    with col2:
        mode_text = "Fast Mode ⚡" if fast_mode else "Standard Mode 🎯"
        st.metric("Training Mode", mode_text)
    with col3:
        pred_text = "Classification" if model_type == "classification" else "Regression"
        st.metric("Prediction Type", pred_text)
    
    # Display results
    if model_type == "classification":
        results_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'Accuracy': [results['Accuracy'] for results in model_results.values()],
            'F1 Score': [results['F1 Score'] for results in model_results.values()],
            'CV Accuracy Mean': [results['CV Accuracy Mean'] for results in model_results.values()],
            'CV Accuracy Std': [results['CV Accuracy Std'] for results in model_results.values()]
        })
        best_metric = 'Accuracy'
    else:
        results_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'RMSE': [results['RMSE'] for results in model_results.values()],
            'R² Score': [results['R² Score'] for results in model_results.values()],
            'MAE': [results['MAE'] for results in model_results.values()],
            'CV R² Mean': [results['CV R² Mean'] for results in model_results.values()],
            'CV R² Std': [results['CV R² Std'] for results in model_results.values()]
        })
        best_metric = 'R² Score'
    
    # Enhanced results display
    st.subheader("📊 Model Performance Results")
    
    # Round results and apply styling
    results_df_rounded = results_df.round(4)
    
    if model_type == "classification":
        styled_df = results_df_rounded.style.highlight_max(subset=['Accuracy', 'F1 Score'], color='lightgreen')
    else:
        styled_df = results_df_rounded.style.highlight_max(subset=['R² Score'], color='lightgreen')
        styled_df = styled_df.highlight_min(subset=['RMSE', 'MAE'], color='lightblue')
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Best model identification
    if model_type == "classification":
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_score = results_df['Accuracy'].max()
        st.success(f"🏆 **Best Model: {best_model_name}** (Accuracy: {best_score:.4f})")
    else:
        best_model_name = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
        best_score = results_df['R² Score'].max()
        st.success(f"🏆 **Best Model: {best_model_name}** (R² Score: {best_score:.4f})")
    
    # Model comparison visualization
    col1, col2 = st.columns(2)
    
    with col1:
        if model_type == "classification":
            fig_metric1 = px.bar(
                results_df, x='Model', y='Accuracy',
                title="Model Accuracy Comparison",
                color='Accuracy',
                color_continuous_scale='viridis'
            )
        else:
            fig_metric1 = px.bar(
                results_df, x='Model', y='R² Score',
                title="Model R² Score Comparison",
                color='R² Score',
                color_continuous_scale='viridis'
            )
        st.plotly_chart(fig_metric1, use_container_width=True)
    
    with col2:
        if model_type == "classification":
            fig_metric2 = px.bar(
                results_df, x='Model', y='F1 Score',
                title="Model F1 Score Comparison",
                color='F1 Score',
                color_continuous_scale='plasma'
            )
        else:
            fig_metric2 = px.bar(
                results_df, x='Model', y='RMSE',
                title="Model RMSE Comparison",
                color='RMSE',
                color_continuous_scale='plasma'
            )
        st.plotly_chart(fig_metric2, use_container_width=True)
    
    # Store results for interpretation
    st.session_state['best_model'] = model_results[best_model_name]['model']
    st.session_state['best_model_name'] = best_model_name
    st.session_state['encoders'] = encoders
    st.session_state['features'] = features
    st.session_state['model_results'] = model_results
    st.session_state['model_type'] = model_type

def model_interpretation_section(df):
    """Model Interpretation Section"""
    st.header("🔍 Model Interpretation")
    
    if 'best_model' not in st.session_state:
        st.warning("Please run the Model Application section first!")
        return
    
    best_model = st.session_state['best_model']
    best_model_name = st.session_state['best_model_name']
    features = st.session_state['features']
    model_type = st.session_state['model_type']
    
    st.subheader(f"📊 {best_model_name} Model Interpretation")
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            importance_df, x='Importance', y='Feature',
            title=f"Feature Importance - {best_model_name}",
            orientation='h',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Insights
        st.write("**Key Insights:**")
        st.write(f"• Most important feature: **{importance_df.iloc[0]['Feature']}** ({importance_df.iloc[0]['Importance']:.3f})")
        st.write(f"• Second most important: **{importance_df.iloc[1]['Feature']}** ({importance_df.iloc[1]['Importance']:.3f})")
        st.write(f"• Least important: **{importance_df.iloc[-1]['Feature']}** ({importance_df.iloc[-1]['Importance']:.3f})")
    
    # Match prediction tool
    st.subheader("⚽ Match Prediction Tool")
    
    # Get unique teams, tournaments, etc. for the prediction interface
    all_teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    all_teams = [team for team in all_teams if team != 'Unknown']
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", all_teams, key="home_pred")
        away_team = st.selectbox("Away Team", all_teams, key="away_pred")
        tournament = st.selectbox("Tournament", sorted(df['tournament'].unique()), key="tournament_pred")
        
    with col2:
        country = st.selectbox("Country", sorted(df['country'].unique()), key="country_pred")
        neutral = st.checkbox("Neutral Venue", key="neutral_pred")
        month = st.selectbox("Month", list(range(1, 13)), key="month_pred")
        
        if st.button("🔮 Predict Match", type="primary"):
            try:
                # Encode the inputs using the same encoders
                encoders = st.session_state['encoders']
                team_strength = encoders['team_strength']
                
                # Get team strengths
                home_strength = team_strength.get(home_team, 0.33)
                away_strength = team_strength.get(away_team, 0.33)
                strength_diff = home_strength - away_strength
                
                # Create feature array
                input_features = np.array([[
                    hash(home_team) % 1000,  # Simple encoding for demo
                    hash(away_team) % 1000,
                    hash(tournament) % 100,
                    hash(country) % 100,
                    int(neutral),
                    month,
                    home_strength,
                    away_strength,
                    strength_diff
                ]])
                
                # Scale features
                input_scaled = encoders['scaler'].transform(input_features)
                
                # Predict
                if model_type == "classification":
                    prediction = best_model.predict(input_scaled)[0]
                    probabilities = best_model.predict_proba(input_scaled)[0]
                    
                    # Decode prediction
                    label_encoder = encoders['label_encoder']
                    outcome_labels = label_encoder.classes_
                    predicted_outcome = label_encoder.inverse_transform([prediction])[0]
                    
                    st.success(f"**Predicted Outcome: {predicted_outcome}**")
                    
                    # Show probabilities
                    prob_df = pd.DataFrame({
                        'Outcome': outcome_labels,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    fig_prob = px.bar(
                        prob_df, x='Outcome', y='Probability',
                        title="Prediction Probabilities",
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                else:
                    prediction = best_model.predict(input_scaled)[0]
                    st.success(f"**Predicted Total Goals: {prediction:.1f}**")
                
            except Exception as e:
                st.error(f"Error in prediction: {e}")

def export_section(df):
    """Export Results Section"""
    st.header("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💾 Clean Dataset")
        
        # Prepare clean dataset
        clean_df = df.copy()
        
        # Remove outliers in total goals
        Q1 = clean_df['total_goals'].quantile(0.25)
        Q3 = clean_df['total_goals'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        clean_df = clean_df[(clean_df['total_goals'] >= lower_bound) & (clean_df['total_goals'] <= upper_bound)]
        
        csv_clean = clean_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Clean Dataset",
            data=csv_clean,
            file_name="clean_football_data.csv",
            mime="text/csv"
        )
        
        st.write(f"**Original records:** {len(df):,}")
        st.write(f"**Clean records:** {len(clean_df):,}")
        st.write(f"**Removed outliers:** {len(df) - len(clean_df):,}")
    
    with col2:
        st.subheader("🤖 Best Model")
        
        if 'best_model' in st.session_state:
            model_data = {
                'model': st.session_state['best_model'],
                'model_name': st.session_state['best_model_name'],
                'encoders': st.session_state['encoders'],
                'features': st.session_state['features'],
                'model_results': st.session_state['model_results']
            }
            
            model_buffer = io.BytesIO()
            pickle.dump(model_data, model_buffer)
            model_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Best Model",
                data=model_buffer.getvalue(),
                file_name="best_football_model.pkl",
                mime="application/octet-stream"
            )
            
            st.write(f"**Model:** {st.session_state['best_model_name']}")
            st.write(f"**Type:** {st.session_state['model_type']}")
        else:
            st.warning("Run Model Application first!")
    
    with col3:
        st.subheader("📊 Analysis Report")
        
        # Generate summary report
        total_teams = len(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        avg_goals = df['total_goals'].mean()
        home_win_pct = (df['outcome'] == 'Home Win').mean() * 100
        
        report = f"""
# Football Analysis Report

## Dataset Overview
- Total Matches: {len(df):,}
- Unique Teams: {total_teams:,}
- Tournaments: {df['tournament'].nunique()}
- Countries: {df['country'].nunique()}
- Time Period: {df['year'].min()} - {df['year'].max()}

## Match Statistics
- Average Goals per Match: {avg_goals:.2f}
- Total Goals Scored: {df['total_goals'].sum():,}
- Home Win Percentage: {home_win_pct:.1f}%
- Draw Percentage: {((df['outcome'] == 'Draw').mean() * 100):.1f}%
- Away Win Percentage: {((df['outcome'] == 'Away Win').mean() * 100):.1f}%

## Key Insights
- Most Common Score: {df['home_score'].mode().iloc[0]}-{df['away_score'].mode().iloc[0]}
- Highest Scoring Match: {df['total_goals'].max()} goals
- Most Active Tournament: {df['tournament'].value_counts().index[0]}
- Most Matches in: {df['country'].value_counts().index[0]}

## Model Performance
"""
        
        if 'model_results' in st.session_state:
            for name, results in st.session_state['model_results'].items():
                if st.session_state['model_type'] == 'classification':
                    report += f"- {name}: Accuracy = {results['Accuracy']:.4f}\n"
                else:
                    report += f"- {name}: R² = {results['R² Score']:.4f}\n"
        
        report += f"""
## Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="football_analysis_report.md",
            mime="text/markdown"
        )

def main():
    """Main application function"""
    st.title("⚽ Football Analytics Dashboard")
    st.markdown("### Comprehensive Analysis of Football Match Data (1872-2025)")
    
    # Performance optimization notice
    with st.expander("⚡ Performance Optimizations", expanded=False):
        st.markdown("""
        **This app includes several performance optimizations for faster results:**
        
        🚀 **Model Training Optimizations:**
        - Smart data sampling (15,000 rows max for ML models)
        - Fast mode with reduced estimators (50 vs 100 trees)
        - Parallel processing with multiple CPU cores
        - Simplified cross-validation in fast mode
        
        📊 **Visualization Optimizations:**
        - Top tournaments/teams filtering for better visualization
        - Efficient data aggregation and caching
        - Modern timeline view (1950+ for clarity)
        
        💡 **Pro Tips:**
        - Use "Fast Mode" in Model Application for quick results
        - Apply date/team filters to reduce dataset size
        - Focus on specific tournaments for detailed analysis
        """)
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check if data.csv exists in the current directory.")
        return
    
    # Create sidebar filters
    filters = create_sidebar_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Show filter results
    if len(filtered_df) != len(df):
        st.info(f"Showing {len(filtered_df):,} matches out of {len(df):,} total matches based on your filters.")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.header("📑 Navigation")
    
    page = st.sidebar.radio(
        "Select Section",
        ["Data Overview", "EDA", "Visualizations", "Model Application", "Model Interpretation", "Export Results"]
    )
    
    # Main content based on selection
    if page == "Data Overview":
        data_overview_section(filtered_df)
    elif page == "EDA":
        eda_section(filtered_df)
    elif page == "Visualizations":
        visualization_section(filtered_df)
    elif page == "Model Application":
        model_application_section(filtered_df)
    elif page == "Model Interpretation":
        model_interpretation_section(filtered_df)
    elif page == "Export Results":
        export_section(filtered_df)

if __name__ == "__main__":
    main()