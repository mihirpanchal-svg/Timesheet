import pandas as pd
import streamlit as st
from datetime import datetime, time, timedelta
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

# Configure page
st.set_page_config(
    page_title="AI-Driven Timesheet Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .ai-feature-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ff4757;
    }
    
    .anomaly-box {
        background: #ffe0e0;
        border: 1px solid #ffb3b3;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #cc0000;
    }
    
    .prediction-box {
        background: #e0f0ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0066cc;
    }
    
    .filter-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def smart_column_mapping(columns):
    """AI-powered column detection - maps similar column names automatically"""
    mapping_rules = {
        'date': ['date', 'day', 'work_date', 'workdate', 'timestamp', 'when', 'fecha'],
        'developer': ['developer','resource', 'dev', 'name', 'employee', 'worker', 'person', 'user', 'programmer','Members','members','Resource'],
        'categorization': ['activity','categorization', 'category', 'type', 'status', 'classification', 'tag', 'categoria','activity description'],
        'time': ['time', 'hours', 'duration', 'work_time', 'worktime', 'effort', 'spent', 'horas']
    }

    detected_mapping = {}
    columns_lower = [col.lower().strip() for col in columns]

    for standard_name, variations in mapping_rules.items():
        for variation in variations:
            matches = [i for i, col in enumerate(columns_lower) if variation in col]
            if matches:
                detected_mapping[standard_name] = columns[matches[0]]
                break

    return detected_mapping

def ai_anomaly_detection(df):
    """AI-powered anomaly detection using Isolation Forest with smart column mapping"""
    if df.empty or len(df) < 10:
        return [], "❌ Insufficient data for anomaly detection (minimum 10 rows required)"
    
    try:
        # Smart column mapping
        column_map = smart_column_mapping(df.columns)
        required_cols = ['date', 'developer', 'categorization', 'time']
        for col in required_cols:
            if col not in column_map:
                return [], f"❌ Missing required column: {col.capitalize()}"
        
        # Prepare features
        features_df = df.copy()
        features_df['TimeHours'] = features_df[column_map['time']].apply(excel_time_to_hours)
        
        # Parse dates
        features_df['ParsedDate'] = pd.to_datetime(features_df[column_map['date']], errors='coerce')
        features_df = features_df.dropna(subset=['ParsedDate'])
        
        # Additional features
        features_df['DayOfWeek'] = features_df['ParsedDate'].dt.dayofweek
        features_df['WeekOfYear'] = features_df['ParsedDate'].dt.isocalendar().week
        
        # Encode categorical columns
        dev_encoded = pd.get_dummies(features_df[column_map['developer']], prefix='dev').astype(int)
        cat_encoded = pd.get_dummies(features_df[column_map['categorization']], prefix='cat').astype(int)
        
        # Combine features
        feature_matrix = pd.concat([
            features_df[['TimeHours', 'DayOfWeek', 'WeekOfYear']],
            dev_encoded,
            cat_encoded
        ], axis=1).fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Isolation Forest
        isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = isolation_forest.fit_predict(features_scaled)
        anomaly_scores = isolation_forest.decision_function(features_scaled)
        
        anomalies = []
        for idx, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
            if label == -1:
                anomaly_data = {
                    'index': idx,
                    'date': features_df.iloc[idx][column_map['date']],
                    'developer': features_df.iloc[idx][column_map['developer']],
                    'category': features_df.iloc[idx][column_map['categorization']],
                    'hours': features_df.iloc[idx]['TimeHours'],
                    'anomaly_score': score,
                    'severity': 'High' if score < -0.5 else 'Medium' if score < -0.2 else 'Low'
                }
                anomalies.append(anomaly_data)
        
        anomalies.sort(key=lambda x: x['anomaly_score'])
        return anomalies, None

    except Exception as e:
        return [], f"❌ Error in anomaly detection: {str(e)}"

def predict_future_hours(daily_data, future_days=30):
    """Predict future hours using simple ML (Linear Regression)"""
    if len(daily_data) < 5:
        return 0.0  # not enough data

    daily_data = daily_data.sort_values('Date')
    daily_data['DayOfWeek'] = daily_data['Date'].dt.dayofweek
    daily_data['PrevHours'] = daily_data['Hours'].shift(1)
    daily_data['RollingAvg'] = daily_data['Hours'].rolling(window=3, min_periods=1).mean().shift(1)
    daily_data = daily_data.dropna()

    if len(daily_data) < 3:
        return 0.0

    X = daily_data[['DayOfWeek', 'PrevHours', 'RollingAvg']]
    y = daily_data['Hours']

    model = LinearRegression()
    model.fit(X, y)

    # Use the last known values to start predicting
    last_date = daily_data.iloc[-1]['Date']
    last_hours = daily_data.iloc[-1]['Hours']
    rolling_avg = daily_data['Hours'].rolling(window=3).mean().iloc[-1]

    total_pred = 0
    for i in range(future_days):
        next_day = last_date + pd.Timedelta(days=1)
        day_of_week = next_day.dayofweek

        features = [[day_of_week, last_hours, rolling_avg]]
        pred = model.predict(features)[0]
        pred = max(pred, 0)  # no negative hours

        total_pred += pred

        # Update for next iteration
        last_date = next_day
        last_hours = pred
        rolling_avg = (rolling_avg * 2 + pred) / 3  # pseudo rolling

    return round(total_pred, 2)

def ai_predictive_analytics(df):
    """AI-powered predictive analytics using ML for next-day workload"""
    if df.empty or len(df) < 5:
        return {}, "❌ Insufficient data for predictions (minimum 5 rows required)"

    try:
        column_map = smart_column_mapping(df.columns)
        required_cols = ['date', 'developer', 'time', 'categorization']
        for col in required_cols:
            if col not in column_map:
                return {}, f"❌ Missing required column: {col.capitalize()}"

        df = df.copy()
        df[column_map['date']] = pd.to_datetime(df[column_map['date']], errors='coerce')
        df = df.dropna(subset=[column_map['date']])
        df['TimeHours'] = df[column_map['time']].apply(excel_time_to_hours)
        df = df[df['TimeHours'] > 0]

        if df.empty:
            return {}, "❌ No valid time entries found"

        predictions = {}
        for dev in df[column_map['developer']].unique():
            dev_df = df[df[column_map['developer']] == dev].sort_values(column_map['date'])

            if len(dev_df) < 5:
                continue

            # Feature Engineering
            daily = dev_df.groupby(column_map['date'])['TimeHours'].sum().reset_index()
            daily['DayOfWeek'] = daily[column_map['date']].dt.dayofweek
            daily['PrevHours'] = daily['TimeHours'].shift(1)
            daily['RollingAvg'] = daily['TimeHours'].rolling(window=3, min_periods=1).mean().shift(1)
            daily = daily.dropna()

            if len(daily) < 3:
                continue

            X = daily[['DayOfWeek', 'PrevHours', 'RollingAvg']]
            y = daily['TimeHours']

            model = LinearRegression()
            model.fit(X, y)

            latest = daily.iloc[-1]
            next_features = [[
                (latest[column_map['date']] + pd.Timedelta(days=1)).dayofweek,
                latest['TimeHours'],
                latest['RollingAvg']
            ]]

            predicted_daily_hours = model.predict(next_features)[0]

            # Total / Avg / Max / Min
            all_data = daily['TimeHours']
            avg_hours = all_data.mean()

            # Trend
            mid = len(all_data) // 2
            first_avg = all_data.iloc[:mid].mean() if mid > 0 else 0
            second_avg = all_data.iloc[mid:].mean() if mid > 0 else 0
            trend_percentage = round(((second_avg - first_avg) / first_avg) * 100, 1) if first_avg > 0 else 0.0

            # Consistency Score
            std_dev = all_data.std()
            consistency_score = max(0, min(1, 1 - (std_dev / avg_hours))) if avg_hours > 0 else 0

            # Burnout Risk
            high_workload_days = (all_data > 10).sum()
            very_high_days = (all_data > 12).sum()
            base_risk = (high_workload_days / len(all_data)) * 50
            extreme_risk = (very_high_days / len(all_data)) * 30
            consistency_risk = (1 - consistency_score) * 20
            burnout_risk = min(100, base_risk + extreme_risk + consistency_risk)

            # Productive / Non-Productive
            dev_cat = dev_df[[column_map['categorization'], 'TimeHours']].copy()
            dev_cat['is_productive'] = dev_cat[column_map['categorization']].str.lower().str.strip().isin(['development', 'testing'])
            total_productive_hours = dev_cat[dev_cat['is_productive']]['TimeHours'].sum()
            total_non_productive_hours = dev_cat[~dev_cat['is_productive']]['TimeHours'].sum()
            predicted_30d_productive_hours = round(predicted_daily_hours * 20 * (total_productive_hours / all_data.sum()), 2) if all_data.sum() > 0 else 0
            predicted_30d_non_productive_hours = round(predicted_daily_hours * 20 * (total_non_productive_hours / all_data.sum()), 2) if all_data.sum() > 0 else 0
            
            predictions[dev] = {
                'predicted_daily_hours': round(predicted_daily_hours, 2),
                'total_days': len(all_data),
                'avg_hours': round(avg_hours, 2),
                'max_daily_hours': round(all_data.max(), 2),
                'min_daily_hours': round(all_data.min(), 2),
                'trend_percentage': trend_percentage,
                'consistency_score': round(consistency_score, 3),
                'burnout_risk': round(burnout_risk, 1),
                'total_productive_hours': round(total_productive_hours, 2),
                'total_non_productive_hours': round(total_non_productive_hours, 2),
                'predicted_30d_productive_hours': predicted_30d_productive_hours,
                'predicted_30d_non_productive_hours': predicted_30d_non_productive_hours
            }

        if not predictions:
            return {}, "❌ No predictions could be made (insufficient developer data)"
        return predictions, None

    except Exception as e:
        return {}, f"❌ Error in ML predictions: {str(e)}"

def ai_data_quality_checker(df):
    """AI-powered data quality analysis with smart column mapping"""
    issues = []
    suggestions = []

    # Smart column mapping
    column_map = smart_column_mapping(df.columns)

    # Check for missing data
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        issues.append(f"❗ Issue detected: {missing_data[missing_data > 0].to_dict()}")
        suggestions.append("🔧 Consider filling missing values or removing incomplete records")

    # Time anomalies
    if 'time' in column_map:
        time_values = pd.to_numeric(df[column_map['time']], errors='coerce')
        if time_values.max() > 24:
            issues.append(f"⏱️ Unusually high time values detected (max: {time_values.max():.1f} hours)")
            suggestions.append("⚠️ Review entries with >16 hours - possible data entry errors")

    # Date range check
    if 'date' in column_map:
        parsed_dates = pd.to_datetime(df[column_map['date']], errors='coerce')
        if parsed_dates.notna().any():
            span_days = (parsed_dates.max() - parsed_dates.min()).days
            if span_days > 365:
                issues.append(f"📅 Data spans {span_days} days - very long time period")
                suggestions.append("📊 Consider analyzing data in smaller chunks (e.g., quarterly/monthly)")

    # Duplicate check
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        issues.append(f"📋 Found {num_duplicates} duplicate entries")
        suggestions.append("🧹 Remove duplicate entries to avoid double-counting")

    return issues, suggestions

def hours_to_hhmm(hours):
    """Convert decimal hours to HH:MM format"""
    if pd.isna(hours) or hours == 0:
        return "00:00"

    total_hours = int(hours)
    total_minutes = int((hours - total_hours) * 60)
    return f"{total_hours:02d}:{total_minutes:02d}"

def hhmm_to_hours(time_str):
    """Convert HH:MM format to decimal hours"""
    if pd.isna(time_str) or time_str == "00:00":
        return 0.0
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes / 60
    except:
        return 0.0

def excel_time_to_hours(t):
    """Enhanced time parsing function"""
    try:
        if pd.isna(t):
            return 0.0

        if isinstance(t, pd.Timestamp):
            return t.hour + t.minute / 60 + t.second / 3600
        elif isinstance(t, datetime):
            return t.hour + t.minute / 60 + t.second / 3600
        elif isinstance(t, time):
            return t.hour + t.minute / 60 + t.second / 3600
        elif isinstance(t, timedelta):
            return t.total_seconds() / 3600
        elif isinstance(t, (int, float)):
            if 0 <= t < 1.0:
                return float(t) * 24
            elif 1 <= t <= 24:
                return float(t)
            else:
                return 0.0
        elif isinstance(t, str):
            t = t.strip()
            if ':' in t:
                try:
                    parts = t.split(':')
                    hours = float(parts[0])
                    minutes = float(parts[1]) if len(parts) > 1 else 0
                    seconds = float(parts[2]) if len(parts) > 2 else 0
                    return hours + minutes/60 + seconds/3600
                except (ValueError, IndexError):
                    pass
            
            dt_obj = pd.to_datetime(t, errors='coerce')
            if pd.notna(dt_obj):
                return dt_obj.hour + dt_obj.minute / 60 + dt_obj.second / 3600
            else:
                return 0.0
        else:
            return 0.0
    except Exception:
        return 0.0

def process_timesheet_data(df, selected_categories=None, selected_users=None):
    """Process timesheet data with category and user filtering"""
    # AI-powered column mapping
    detected_columns = smart_column_mapping(df.columns.tolist())
    
    # Map detected columns to standard names
    column_mapping = {}
    for standard, detected in detected_columns.items():
        if standard == 'date':
            column_mapping['Date'] = detected
        elif standard == 'developer':
            column_mapping['Developer'] = detected
        elif standard == 'categorization':
            column_mapping['Categorization'] = detected
        elif standard == 'time':
            column_mapping['Time'] = detected

    # Rename columns if we detected alternatives
    if column_mapping:
        df = df.rename(columns={v: k for k, v in column_mapping.items()})

    # Check for required columns
    required_cols = ['Date', 'Developer', 'Categorization', 'Time']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notnull()]
    
    if df.empty:
        return None, "Error: No valid dates found"

    # Filter by selected categories
    if selected_categories:
        category_mask = df['Categorization'].astype(str).str.strip().str.lower().isin(
            [cat.lower() for cat in selected_categories]
        )
        df = df[category_mask]
    
    # Filter by selected users/developers
    if selected_users:
        user_mask = df['Developer'].astype(str).str.strip().isin(selected_users)
        df = df[user_mask]

    if df.empty:
        return None, "No data found for selected categories and users"

    # Add week start column
    df['WeekStart'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    # Convert time to hours
    df['TimeHours'] = df['Time'].apply(excel_time_to_hours)

    # Group by developer and week
    result = (
        df.groupby(['Developer', 'WeekStart'])['TimeHours']
        .sum()
        .reset_index()
    )

    result['WeekStart'] = result['WeekStart'].dt.strftime('%Y-%m-%d')
    result.rename(columns={'WeekStart': 'Week Start'}, inplace=True)
    result['Total Hours'] = result['TimeHours'].apply(hours_to_hhmm)
    result = result.drop('TimeHours', axis=1)

    return result, None

def create_visualizations(df, original_df):
    """Create interactive visualizations"""
    # Convert back to decimal for calculations
    df_calc = df.copy()
    df_calc['Hours_Decimal'] = df_calc['Total Hours'].apply(hhmm_to_hours)
    
    # 1. Developer productivity chart
    dev_totals = df_calc.groupby('Developer')['Hours_Decimal'].sum().sort_values(ascending=True)
    
    fig1 = px.bar(
        x=dev_totals.values,
        y=dev_totals.index,
        orientation='h',
        title="Total Hours by Developer",
        labels={'x': 'Hours', 'y': 'Developer'},
        color=dev_totals.values,
        color_continuous_scale='viridis'
    )
    fig1.update_layout(height=400)
    
    # 2. Weekly trend
    weekly_totals = df_calc.groupby('Week Start')['Hours_Decimal'].sum().reset_index()
    weekly_totals['Week Start'] = pd.to_datetime(weekly_totals['Week Start'])
    
    fig2 = px.line(
        weekly_totals,
        x='Week Start',
        y='Hours_Decimal',
        title="Weekly Hours Trend",
        markers=True
    )
    fig2.update_layout(height=400)
    
    # 3. Category distribution (if original data available)
    if original_df is not None:
        column_map = smart_column_mapping(original_df.columns)

        if 'categorization' in column_map:
            cat_col = column_map['categorization']
            
            category_hours = original_df.groupby(cat_col).size().reset_index(name='Count')
            
            fig3 = px.pie(
                category_hours,
                values='Count',
                names=cat_col,
                title="📊 Task Distribution by Category"
            )
            fig3.update_layout(height=400)
        else:
            fig3 = None
    else:
        fig3 = None

    return fig1, fig2, fig3

def generate_insights(df, original_df):
    """Generate AI insights"""
    insights = []
    
    if df.empty:
        return ["No data available for analysis"]
    
    # Convert to decimal for analysis
    df_calc = df.copy()
    df_calc['Hours_Decimal'] = df_calc['Total Hours'].apply(hhmm_to_hours)
    
    # Developer analysis
    dev_hours = df_calc.groupby('Developer')['Hours_Decimal'].sum().sort_values(ascending=False)
    total_devs = len(dev_hours)
    
    insights.append(f"📊 PRODUCTIVITY ANALYSIS ({total_devs} developers)")
    insights.append(f" Top performer: {dev_hours.index[0]} ({hours_to_hhmm(dev_hours.iloc[0])})")
    insights.append(f" Average hours per developer: {hours_to_hhmm(dev_hours.mean())}")
    
    # Workload distribution
    std_dev = dev_hours.std()
    mean_hours = dev_hours.mean()
    if std_dev > mean_hours * 0.3:
        insights.append("⚠️ HIGH WORKLOAD VARIATION** detected - consider redistributing tasks")
    
    # Weekly patterns
    weekly_totals = df_calc.groupby('Week Start')['Hours_Decimal'].sum()
    if len(weekly_totals) > 1:
        insights.append(f"📅 WEEKLY PATTERNS:")
        insights.append(f" Most productive week: {weekly_totals.idxmax()} ({hours_to_hhmm(weekly_totals.max())})")
        insights.append(f" Average weekly hours: {hours_to_hhmm(weekly_totals.mean())}")
    
    # Efficiency recommendations
    low_performers = dev_hours[dev_hours < dev_hours.mean() * 0.7]
    if len(low_performers) > 0:
        insights.append(f"🎯 RECOMMENDATIONS:")
        insights.append(f"• {len(low_performers)} developers below 70% of average - may need support")
    
    return insights

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Driven Timesheet Analyzer</h1>
        <p>Intelligent timesheet analysis with AI anomaly detection and predictive insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload & Settings")
        
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            help="Upload your timesheet Excel file"
        )
        
        if uploaded_file is not None:
            # Load and process data
            try:
                df = pd.read_excel(uploaded_file)
                df = df.dropna(how='all')
                
                # Display basic info
                st.success(f"✅ File loaded successfully!")
                st.info(f"📊 **{len(df)} records** across **{len(df.columns)} columns**")
                
                # Show column mapping
                detected_columns = smart_column_mapping(df.columns.tolist())
                if detected_columns:
                    st.subheader("🔍 Detected Columns")
                    for standard, detected in detected_columns.items():
                        st.write(f"• {standard.title()}: `{detected}`")
                
                # Get unique categories and users
                categorization_col = detected_columns.get('categorization', 'Categorization')
                developer_col = detected_columns.get('developer', 'Developer')
                
                if categorization_col in df.columns and developer_col in df.columns:
                    unique_categories = df[categorization_col].dropna().unique().tolist()
                    unique_users = df[developer_col].dropna().unique().tolist()
                    
                    # Category Filter
                    st.subheader("📂 Category Filter")
                    selected_categories = st.multiselect(
                        "Select categories to analyze:",
                        options=unique_categories,
                        default=unique_categories,
                        help="Select one or more categories to filter the analysis"
                    )
                    
                    # User/Developer Filter
                    st.subheader("👤 User Filter")
                    selected_users = st.multiselect(
                        "Select users/developers to analyze:",
                        options=unique_users,
                        default=unique_users,
                        help="Select one or more users to filter the analysis"
                    )
                    
                    # Show filter summary
                    if selected_categories and selected_users:
                        st.markdown(f"""
                        <div class="filter-container">
                            <strong>🎯 Active Filters:</strong><br>
                            📂 Categories: {len(selected_categories)} of {len(unique_categories)}<br>
                            👤 Users: {len(selected_users)} of {len(unique_users)}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AI Features Toggle
                    st.subheader("🤖 AI Features")
                    enable_anomaly_detection = st.checkbox("Enable Anomaly Detection", value=False)
                    enable_predictive_analytics = st.checkbox("Enable Predictive Analytics", value=False)
                    
                    # Update results in real-time
                    if selected_categories and selected_users:
                        result_df, error = process_timesheet_data(df, selected_categories, selected_users)
                        
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            # Store in session state
                            st.session_state.result_df = result_df
                            st.session_state.original_df = df
                            st.session_state.selected_categories = selected_categories
                            st.session_state.selected_users = selected_users
                            st.session_state.enable_anomaly_detection = enable_anomaly_detection
                            st.session_state.enable_predictive_analytics = enable_predictive_analytics
                else:
                    st.error("❌ Required columns not found!")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Main content area
    if 'result_df' in st.session_state:
        result_df = st.session_state.result_df
        original_df = st.session_state.original_df
        selected_categories = st.session_state.selected_categories
        selected_users = st.session_state.selected_users
        enable_anomaly_detection = st.session_state.get('enable_anomaly_detection', True)
        enable_predictive_analytics = st.session_state.get('enable_predictive_analytics', True)
        
        # Show selected filters
        st.subheader("🎯 Applied Filters")
        
        # Create columns for filters display
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            st.markdown("### 📂 Selected Categories")
            if len(selected_categories) <= 3:
                for cat in selected_categories:
                    st.markdown(f"""
                    <div class="metric-container">
                        <strong>{cat}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <strong>{len(selected_categories)} categories selected</strong><br>
                    <small>{', '.join(selected_categories[:3])}{'...' if len(selected_categories) > 3 else ''}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with filter_col2:
            st.markdown("### 👤 Selected Users")
            if len(selected_users) <= 3:
                for user in selected_users:
                    st.markdown(f"""
                    <div class="metric-container">
                        <strong>{user}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <strong>{len(selected_users)} users selected</strong><br>
                    <small>{', '.join(selected_users[:3])}{'...' if len(selected_users) > 3 else ''}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Filter the original dataframe for AI features
        filtered_original_df = original_df.copy()
        detected_columns = smart_column_mapping(original_df.columns)
        
        if 'categorization' in detected_columns:
            cat_col = detected_columns['categorization']
            category_mask = filtered_original_df[cat_col].astype(str).str.strip().str.lower().isin(
                [cat.lower() for cat in selected_categories]
            )
            filtered_original_df = filtered_original_df[category_mask]
        
        if 'developer' in detected_columns:
            dev_col = detected_columns['developer']
            user_mask = filtered_original_df[dev_col].astype(str).str.strip().isin(selected_users)
            filtered_original_df = filtered_original_df[user_mask]
        
        # AI Feature 1: Anomaly Detection
        if enable_anomaly_detection:
            st.subheader("🚨 AI Anomaly Detection")
            
            with st.spinner("🤖 Analyzing filtered data for anomalies..."):
                anomalies, error = ai_anomaly_detection(filtered_original_df)
                
                if error:
                    st.error(f"❌ {error}")
                elif anomalies:
                    st.markdown(f"""
                    <div class="ai-feature-box">
                        <h4>🔍 Detected {len(anomalies)} anomalies in your filtered timesheet data</h4>
                        <p>These entries show unusual patterns that may need attention:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, anomaly in enumerate(anomalies[:5]):  # Show top 5
                        severity_emoji = "🔴" if anomaly['severity'] == 'High' else "🟡" if anomaly['severity'] == 'Medium' else "🟢"
                        st.markdown(f"""
                        <div class="anomaly-box">
                            <strong>{severity_emoji} {anomaly['severity']} Severity Anomaly</strong><br>
                            📅 Date: {anomaly['date']}<br>
                            👤 Developer: {anomaly['developer']}<br>
                            📂 Category: {anomaly['category']}<br>
                            ⏱️ Hours: {anomaly['hours']:.2f}<br>
                            📊 Anomaly Score: {anomaly['anomaly_score']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No anomalies detected in filtered data! Your timesheet data looks normal.")
        
        # AI Feature 2: Predictive Analytics
        if enable_predictive_analytics:
            st.subheader("🔮 AI Predictive Analytics")
            
            with st.spinner("🤖 Generating predictions for filtered data..."):
                predictions, error = ai_predictive_analytics(filtered_original_df)
                
                if error:
                    st.error(f"❌ {error}")
                elif predictions and len(predictions) > 0:
                    st.markdown(f"""
                    <div class="ai-feature-box">
                        <h4>📈 Workload Predictions for {len(predictions)} filtered developers</h4>
                        <p>AI-powered insights about future performance and burnout risk:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a more robust display with better formatting
                    num_cols = min(3, len(predictions))
                    cols = st.columns(num_cols)
                    
                    for i, (dev, pred) in enumerate(predictions.items()):
                        with cols[i % num_cols]:
                            risk_color = "🔴" if pred['burnout_risk'] > 70 else "🟡" if pred['burnout_risk'] > 40 else "🟢"
                            trend_arrow = "📈" if pred['trend_percentage'] > 10 else "📉" if pred['trend_percentage'] < -10 else "➡️"
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <strong>👤 {dev}</strong><br>
                                🔮 Predicted Daily Hours: {pred['predicted_daily_hours']:.1f}<br>
                                {trend_arrow} Trend: {pred['trend_percentage']:.1f}%<br>
                                📊 Consistency: {pred['consistency_score']:.2f}<br>
                                {risk_color} Burnout Risk: {pred['burnout_risk']:.1f}%<br>
                                📅 Data Days: {pred['total_days']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Add a detailed table view
                    st.subheader("📊 Detailed Predictions Table")
                    pred_df = pd.DataFrame(predictions).T
                    pred_df.index.name = 'Developer'
                    pred_df['predicted_daily_hours'] = pred_df['predicted_daily_hours'].round(2)
                    pred_df['trend_percentage'] = pred_df['trend_percentage'].round(1)
                    pred_df['consistency_score'] = pred_df['consistency_score'].round(3)
                    pred_df['burnout_risk'] = pred_df['burnout_risk'].round(1)
                    pred_df['avg_hours'] = pred_df['avg_hours'].round(2)
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                else:
                    st.warning("⚠️ No predictions available for filtered data. This could be due to:")
                    st.write("• Insufficient data (need at least 2 weeks)")
                    st.write("• Data quality issues")
                    st.write("• Limited developer activity in selected filters")
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Weekly Hours Summary (Filtered)")
            st.dataframe(
                result_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = result_df.to_csv(index=False)
            filter_info = f"_cat{len(selected_categories)}_users{len(selected_users)}"
            st.download_button(
                label="📥 Download Filtered CSV",
                data=csv,
                file_name=f"timesheet_analysis_filtered{filter_info}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("📈 Key Metrics")
            
            # Calculate metrics
            df_calc = result_df.copy()
            df_calc['Hours_Decimal'] = df_calc['Total Hours'].apply(hhmm_to_hours)
            
            total_hours = df_calc['Hours_Decimal'].sum()
            avg_weekly = df_calc.groupby('Week Start')['Hours_Decimal'].sum().mean()
            total_devs = df_calc['Developer'].nunique()
            
            st.metric("Total Hours (Filtered)", hours_to_hhmm(total_hours))
            st.metric("Average Weekly", hours_to_hhmm(avg_weekly))
            st.metric("Active Developers", total_devs)
            
            # Additional filter metrics
            st.markdown("### 🎯 Filter Impact")
            original_total_devs = original_df[detected_columns.get('developer', 'Developer')].nunique()
            original_total_categories = original_df[detected_columns.get('categorization', 'Categorization')].nunique()
            
            st.metric("Categories Used", f"{len(selected_categories)}/{original_total_categories}")
            st.metric("Users Analyzed", f"{len(selected_users)}/{original_total_devs}")
        
        # Visualizations
        st.subheader("📊 Analytics Dashboard (Filtered Data)")
        
        fig1, fig2, fig3 = create_visualizations(result_df, filtered_original_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        
        # AI Insights
        st.subheader("🧠 AI Insights (Based on Filtered Data)")
        insights = generate_insights(result_df, filtered_original_df)
        
        for insight in insights:
            st.markdown(f"""
            <div class="insight-box">
                {insight}
            </div>
            """, unsafe_allow_html=True)
        
        # Data Quality Check
        issues, suggestions = ai_data_quality_checker(filtered_original_df)
        
        if issues:
            st.subheader("⚠️ Data Quality Issues (Filtered Data)")
            for issue in issues:
                st.markdown(f"""
                <div class="warning-box">
                    {issue}
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("💡 Suggestions")
            for suggestion in suggestions:
                st.markdown(f"""
                <div class="success-box">
                    {suggestion}
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Welcome message
        st.markdown("""
        <div class="success-box">
            <h3>👋 Welcome to AI-Driven Timesheet Analyzer!</h3>
            <p>Upload your Excel timesheet file to get started with intelligent analysis</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
