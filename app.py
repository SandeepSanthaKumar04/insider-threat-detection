import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.figure_factory as ff
from scipy import stats
import json
import time

# Page configuration with custom theme
st.set_page_config(
    page_title="Enterprise Insider Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add animated loading spinner
def load_with_spinner(message="Loading..."):
    with st.spinner(message):
        time.sleep(0.5)

# Generate more realistic sample data
@st.cache_data
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate timestamps for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate realistic employee behavior patterns
    data = {
        'employee_id': np.arange(1, n_samples + 1),
        'timestamp': timestamps,
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Sales', 'Engineering'], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Department-specific behavior patterns
    dept_patterns = {
        'IT': {
            'access_freq_mean': 70, 'access_freq_std': 15,
            'email_vol_mean': 45, 'email_vol_std': 15,
            'file_dl_mean': 40, 'file_dl_std': 12,
            'after_hours_mean': 6, 'after_hours_std': 2,
            'usb_mean': 3, 'usb_std': 1,
            'print_mean': 3, 'print_std': 1
        },
        'HR': {
            'access_freq_mean': 40, 'access_freq_std': 10,
            'email_vol_mean': 60, 'email_vol_std': 20,
            'file_dl_mean': 25, 'file_dl_std': 8,
            'after_hours_mean': 3, 'after_hours_std': 1,
            'usb_mean': 1, 'usb_std': 0.5,
            'print_mean': 7, 'print_std': 2
        },
        'Finance': {
            'access_freq_mean': 55, 'access_freq_std': 12,
            'email_vol_mean': 50, 'email_vol_std': 15,
            'file_dl_mean': 35, 'file_dl_std': 10,
            'after_hours_mean': 5, 'after_hours_std': 2,
            'usb_mean': 2, 'usb_std': 0.8,
            'print_mean': 6, 'print_std': 2
        },
        'Sales': {
            'access_freq_mean': 35, 'access_freq_std': 10,
            'email_vol_mean': 70, 'email_vol_std': 20,
            'file_dl_mean': 20, 'file_dl_std': 8,
            'after_hours_mean': 4, 'after_hours_std': 2,
            'usb_mean': 2, 'usb_std': 1,
            'print_mean': 4, 'print_std': 1.5
        },
        'Engineering': {
            'access_freq_mean': 60, 'access_freq_std': 15,
            'email_vol_mean': 40, 'email_vol_std': 12,
            'file_dl_mean': 45, 'file_dl_std': 15,
            'after_hours_mean': 5, 'after_hours_std': 2,
            'usb_mean': 4, 'usb_std': 1.5,
            'print_mean': 3, 'print_std': 1
        }
    }
    
    # Generate department-specific metrics
    for dept in df['department'].unique():
        mask = df['department'] == dept
        pattern = dept_patterns[dept]
        
        df.loc[mask, 'access_frequency'] = np.clip(
            np.random.normal(pattern['access_freq_mean'], pattern['access_freq_std'], mask.sum()),
            0, 100)
        df.loc[mask, 'email_volume'] = np.clip(
            np.random.normal(pattern['email_vol_mean'], pattern['email_vol_std'], mask.sum()),
            0, 100)
        df.loc[mask, 'file_downloads'] = np.clip(
            np.random.normal(pattern['file_dl_mean'], pattern['file_dl_std'], mask.sum()),
            0, 100)
        df.loc[mask, 'after_hours_access'] = np.clip(
            np.random.normal(pattern['after_hours_mean'], pattern['after_hours_std'], mask.sum()),
            0, 10)
        df.loc[mask, 'usb_usage'] = np.clip(
            np.random.normal(pattern['usb_mean'], pattern['usb_std'], mask.sum()),
            0, 5)
        df.loc[mask, 'print_jobs'] = np.clip(
            np.random.normal(pattern['print_mean'], pattern['print_std'], mask.sum()),
            0, 10)
    
    # Add other categorical columns
    df['location'] = np.random.choice(['HQ', 'Remote', 'Branch-A', 'Branch-B'], n_samples)
    df['role_level'] = np.random.choice(['Entry', 'Mid', 'Senior', 'Manager'], n_samples)
    
    # Add seasonal patterns
    df['access_frequency'] += np.sin(np.linspace(0, 4*np.pi, n_samples)) * 10
    df['email_volume'] += np.cos(np.linspace(0, 4*np.pi, n_samples)) * 15
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[anomaly_indices, 'access_frequency'] *= 2
    df.loc[anomaly_indices, 'file_downloads'] *= 1.5
    df.loc[anomaly_indices, 'after_hours_access'] *= 3
    
    # Calculate risk score based on multiple factors
    df['risk_score'] = (
        # Normalized access frequency (20% weight)
        (df['access_frequency'] / 100) * 20 +
        
        # Normalized file downloads (30% weight)
        (df['file_downloads'] / 100) * 30 +
        
        # After hours access with diminishing returns (20% weight)
        (np.log1p(df['after_hours_access']) / np.log1p(10)) * 20 +
        
        # USB usage with exponential scaling (15% weight)
        (np.exp(df['usb_usage'] / 5) / np.exp(1)) * 15 +
        
        # Print jobs with square root scaling (15% weight)
        (np.sqrt(df['print_jobs']) / np.sqrt(10)) * 15
    ).clip(0, 100)
    
    # Perform PCA with standardization
    features_for_pca = ['access_frequency', 'email_volume', 'file_downloads', 
                       'after_hours_access', 'usb_usage', 'print_jobs']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features_for_pca])
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_features)
    
    # Store PCA components and explained variance
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    df['PCA3'] = pca_result[:, 2]
    
    # Store the PCA model and feature names for later use
    df.attrs['pca_model'] = pca
    df.attrs['pca_features'] = features_for_pca
    df.attrs['pca_scaler'] = scaler
    
    # Perform Anomaly Detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(scaled_features)
    df['anomaly_score'] = -iso_forest.score_samples(scaled_features)
    
    return df

# Enhanced Custom CSS with animations
st.markdown("""
<style>
    /* ... existing CSS ... */
    
    /* Animation for metric cards */
    .metric-card {
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-in;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    /* Pulse animation for alerts */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .alert-card {
        animation: pulse 2s infinite;
    }
    
    /* Progress bar animation */
    .stProgress > div > div > div > div {
        transition: all 0.3s ease;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: rgba(0,0,0,0.8);
        color: white;
        text-align: center;
        padding: 5px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data
load_with_spinner("Initializing system...")
data = generate_sample_data()

# Sidebar with real-time monitoring
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1);'>
        <h1 style='color: white; font-size: 1.5rem; margin: 0;'>🛡️</h1>
        <h2 style='color: white; font-size: 1.25rem; margin: 0.5rem 0;'>Insider Threat</h2>
        <p style='color: white; opacity: 0.8; margin: 0;'>Detection System</p>
        <div style='margin-top: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 4px;'>
            <p style='color: white; margin: 0;'>System Status: <span style='color: #2ECC71;'>●</span> Active</p>
            <p style='color: white; margin: 0;'>Last Updated: {}</p>
        </div>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# Enhanced Navigation with badges
pages = {
    "Dashboard": {"icon": "📊", "badge": 0},
    "Employee Profiles": {"icon": "👥", "badge": 3},
    "Risk Analysis": {"icon": "⚠️", "badge": 2},
    "Anomaly Detection": {"icon": "🔍", "badge": 5},
    "Trend Analysis": {"icon": "📈", "badge": 0},
    "Advanced Analytics": {"icon": "🧮", "badge": 1}
}

st.sidebar.markdown("""
    <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
        <h3 style='color: white; font-size: 1rem; margin: 0 0 1rem 0;'>📱 Navigation</h3>
    </div>
""", unsafe_allow_html=True)

selected_page = st.sidebar.radio(
    "Navigation Menu",
    list(pages.keys()),
    format_func=lambda x: f"{pages[x]['icon']} {x} {'🔴 ' + str(pages[x]['badge']) if pages[x]['badge'] > 0 else ''}",
    label_visibility="collapsed"
)

if selected_page == "Dashboard":
    # Interactive Dashboard Header
    st.markdown("""
        <div class='dashboard-header'>
            <h1 style='color: #1a365d; margin: 0;'>Enterprise Security Dashboard</h1>
            <p style='color: #4a5568; margin: 0.5rem 0 0 0;'>Real-time monitoring and threat detection</p>
            <div style='display: flex; gap: 1rem; margin-top: 1rem;'>
                <span class='tooltip'>
                    🔄 Last Scan: 2 minutes ago
                    <span class='tooltiptext'>System continuously monitors for threats</span>
                </span>
                <span class='tooltip'>
                    👥 Active Users: 245
                    <span class='tooltiptext'>Currently monitored employees</span>
                </span>
                <span class='tooltip'>
                    ⚡ System Load: 42%
                    <span class='tooltiptext'>Current system resource usage</span>
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Real-time Metrics with Animations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <p class='metric-label'>Active Sessions</p>
                <h3 class='metric-value'>245</h3>
                <p style='margin: 0;'><span style='color: #2ECC71;'>↑ 12</span> from last hour</p>
                <div class='tooltip'>
                    <small>Click for details</small>
                    <span class='tooltiptext'>Active user sessions across all departments</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='alert-card'>
                <p class='metric-label'>Critical Alerts</p>
                <h3 class='metric-value'>5</h3>
                <p style='margin: 0;'><span style='color: #FF4B4B;'>↑ 2</span> new alerts</p>
                <div class='tooltip'>
                    <small>High priority</small>
                    <span class='tooltiptext'>Requires immediate attention</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='success-card'>
                <p class='metric-label'>System Health</p>
                <h3 class='metric-value'>98%</h3>
                <p style='margin: 0;'><span style='color: #2ECC71;'>↑ 3%</span> improvement</p>
                <div class='tooltip'>
                    <small>Optimal</small>
                    <span class='tooltiptext'>System performing efficiently</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='metric-card'>
                <p class='metric-label'>Response Time</p>
                <h3 class='metric-value'>1.2s</h3>
                <p style='margin: 0;'><span style='color: #2ECC71;'>↓ 0.3s</span> faster</p>
                <div class='tooltip'>
                    <small>Performance</small>
                    <span class='tooltiptext'>Average system response time</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Interactive Risk Analysis
    st.markdown("### Risk Analysis Dashboard")
    risk_col1, risk_col2 = st.columns([2, 1])
    
    with risk_col1:
        # Advanced Risk Distribution Chart
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=data['risk_score'],
            name='Risk Distribution',
            nbinsx=30,
            marker_color='rgba(74, 144, 226, 0.7)'
        ))
        
        # Add kernel density estimate
        kde_x = np.linspace(0, 100, 100)
        kde = stats.gaussian_kde(data['risk_score'])
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde(kde_x) * 1000,
            name='Density',
            line=dict(color='rgba(231, 76, 60, 0.8)', width=2)
        ))
        
        fig.update_layout(
            title='Risk Score Distribution with Density Estimation',
            xaxis_title='Risk Score',
            yaxis_title='Frequency',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with risk_col2:
        # Department-wise Risk Analysis
        dept_risk = data.groupby('department')['risk_score'].mean().sort_values(ascending=True)
        
        fig = go.Figure(go.Bar(
            x=dept_risk.values,
            y=dept_risk.index,
            orientation='h',
            marker_color=['#2ECC71' if x < 50 else '#E74C3C' for x in dept_risk.values]
        ))
        
        fig.update_layout(
            title='Department Risk Levels',
            xaxis_title='Average Risk Score',
            yaxis_title='Department',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Interactive Threat Map
    st.markdown("### Geographic Threat Distribution")
    location_risk = data.groupby('location')['risk_score'].agg(['mean', 'count']).round(2)
    
    fig = go.Figure()
    
    # Create bubble chart
    fig.add_trace(go.Scatter(
        x=location_risk.index,
        y=location_risk['mean'],
        mode='markers',
        marker=dict(
            size=location_risk['count'] / 5,
            color=location_risk['mean'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='Risk Score')
        ),
        text=[f"Location: {loc}<br>Risk Score: {risk['mean']}<br>Employee Count: {risk['count']}" 
              for loc, risk in location_risk.iterrows()],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Location-based Risk Assessment',
        xaxis_title='Location',
        yaxis_title='Average Risk Score',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Real-time Activity Feed
    st.markdown("### Live Activity Feed")
    with st.container():
        for i in range(5):
            event_type = np.random.choice(['login', 'download', 'access', 'alert'])
            severity = np.random.choice(['low', 'medium', 'high'])
            time_ago = np.random.randint(1, 60)
            
            color = {
                'low': '#2ECC71',
                'medium': '#F1C40F',
                'high': '#E74C3C'
            }[severity]
            
            st.markdown(f"""
                <div style='padding: 0.5rem; border-left: 4px solid {color}; margin-bottom: 0.5rem; background: white; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>{'🔒' if event_type == 'login' else '📥' if event_type == 'download' else '🔑' if event_type == 'access' else '⚠️'} {event_type.title()} Event</span>
                        <span style='color: #718096;'>{time_ago}m ago</span>
                    </div>
                    <p style='margin: 0.2rem 0 0 0; color: #4A5568;'>User activity detected - Severity: <span style='color: {color};'>{severity.title()}</span></p>
                </div>
            """, unsafe_allow_html=True)

elif selected_page == "Employee Profiles":
    st.markdown("### Employee Profiles")
    
    # Enhanced employee selector with risk indicators
    employees_df = data[['access_frequency', 'email_volume', 'file_downloads', 'risk_score']].copy()
    employees_df['Employee ID'] = employees_df.index
    employees_df['Risk Level'] = pd.cut(employees_df['risk_score'], 
                                      bins=[0, 30, 70, 100], 
                                      labels=['Low', 'Medium', 'High'])
    
    selected_employee = st.selectbox(
        "Select Employee to View",
        employees_df.index,
        format_func=lambda x: f"Employee {x} (Risk: {employees_df.loc[x, 'Risk Level']})"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Activity Metrics")
        
        # Create a radar chart for metrics
        metrics = {
            "Access Frequency": data.iloc[selected_employee]['access_frequency'],
            "Email Volume": data.iloc[selected_employee]['email_volume'],
            "File Downloads": data.iloc[selected_employee]['file_downloads'],
            "After Hours": data.iloc[selected_employee]['after_hours_access'],
            "USB Usage": data.iloc[selected_employee]['usb_usage'],
            "Print Jobs": data.iloc[selected_employee]['print_jobs']
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            line_color='#1a365d'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(metrics.values())]
                )),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk Assessment")
        risk_score = data.iloc[selected_employee]['risk_score']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            delta={'reference': 50},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1a365d"},
                'steps': [
                    {'range': [0, 30], 'color': "#38a169"},
                    {'range': [30, 70], 'color': "#dd6b20"},
                    {'range': [70, 100], 'color': "#e53e3e"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            title_text="Risk Score Gauge",
            title_x=0.5
        )
        
        st.plotly_chart(fig)

elif selected_page == "Risk Analysis":
    st.markdown("### Risk Analysis")
    
    # Enhanced correlation matrix
    st.subheader("Risk Factors Correlation")
    risk_factors = ['access_frequency', 'email_volume', 'file_downloads', 
                    'after_hours_access', 'usb_usage', 'print_jobs']
    
    correlation_matrix = data[risk_factors].corr()
    
    fig = px.imshow(correlation_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu",
                    aspect="auto")
    fig.update_layout(
        title_text="Correlation Heatmap",
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D Risk Analysis
    st.subheader("3D Risk Analysis")
    
    fig = px.scatter_3d(
        data,
        x='access_frequency',
        y='file_downloads',
        z='email_volume',
        color='risk_score',
        color_continuous_scale='Viridis',
        size='risk_score',
        opacity=0.7,
        labels={
            'access_frequency': 'Access Frequency',
            'file_downloads': 'File Downloads',
            'email_volume': 'Email Volume'
        }
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Access Frequency',
            yaxis_title='File Downloads',
            zaxis_title='Email Volume'
        ),
        title_text="3D Risk Factor Analysis",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Risk threshold slider
    risk_threshold = st.slider(
        "Risk Threshold Level",
        0, 100, 70,
        help="Adjust the threshold for risk detection"
    )

elif selected_page == "Anomaly Detection":
    st.markdown("### Anomaly Detection")
    
    # Explanation of anomaly detection
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h4 style='margin-top: 0; color: #000000;'>About Anomaly Detection</h4>
            <p style='color: #000000;'>Our system uses Isolation Forest algorithm to detect anomalies in employee behavior patterns. 
            The algorithm identifies unusual patterns by analyzing multiple factors including:</p>
            <ul style='color: #000000;'>
                <li>Access frequency</li>
                <li>Email volume</li>
                <li>File downloads</li>
                <li>After-hours access</li>
                <li>USB usage</li>
                <li>Print jobs</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Anomaly Statistics
    col1, col2, col3 = st.columns(3)
    
    total_anomalies = len(data[data['anomaly'] == -1])
    anomaly_percentage = (total_anomalies / len(data)) * 100
    
    with col1:
        st.metric(
            "Total Anomalies Detected",
            f"{total_anomalies}",
            f"{anomaly_percentage:.1f}% of total activity"
        )
    
    with col2:
        avg_anomaly_score = data[data['anomaly'] == -1]['anomaly_score'].mean()
        st.metric(
            "Average Anomaly Score",
            f"{avg_anomaly_score:.2f}",
            "Higher score indicates more unusual behavior"
        )
    
    with col3:
        high_risk_anomalies = len(data[(data['anomaly'] == -1) & (data['risk_score'] > 70)])
        st.metric(
            "High Risk Anomalies",
            f"{high_risk_anomalies}",
            f"{(high_risk_anomalies/total_anomalies*100):.1f}% of anomalies"
        )
    
    # 3D Anomaly Visualization
    st.subheader("3D Behavioral Analysis")
    
    # Add visualization controls
    viz_col1, viz_col2 = st.columns([3, 1])
    
    with viz_col2:
        st.markdown("### Visualization Controls")
        point_size = st.slider("Point Size", 5, 50, 20)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7)
        show_normal = st.checkbox("Show Normal Behavior", True)
        show_anomalies = st.checkbox("Show Anomalies", True)
    
    with viz_col1:
        fig = go.Figure()
        
        if show_normal:
            normal_data = data[data['anomaly'] == 1]
            fig.add_trace(go.Scatter3d(
                x=normal_data['PCA1'],
                y=normal_data['PCA2'],
                z=normal_data['PCA3'],
                mode='markers',
                name='Normal Behavior',
                marker=dict(
                    size=point_size,
                    color='#2ECC71',
                    opacity=opacity
                ),
                hovertemplate=(
                    "Access: %{customdata[0]:.0f}<br>" +
                    "Email: %{customdata[1]:.0f}<br>" +
                    "Downloads: %{customdata[2]:.0f}<br>" +
                    "Risk Score: %{customdata[3]:.1f}"
                ),
                customdata=normal_data[['access_frequency', 'email_volume', 
                                      'file_downloads', 'risk_score']]
            ))
        
        if show_anomalies:
            anomaly_data = data[data['anomaly'] == -1]
            fig.add_trace(go.Scatter3d(
                x=anomaly_data['PCA1'],
                y=anomaly_data['PCA2'],
                z=anomaly_data['PCA3'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=point_size * 1.2,
                    color='#E74C3C',
                    opacity=opacity
                ),
                hovertemplate=(
                    "Access: %{customdata[0]:.0f}<br>" +
                    "Email: %{customdata[1]:.0f}<br>" +
                    "Downloads: %{customdata[2]:.0f}<br>" +
                    "Risk Score: %{customdata[3]:.1f}<br>" +
                    "Anomaly Score: %{customdata[4]:.2f}"
                ),
                customdata=anomaly_data[['access_frequency', 'email_volume', 
                                       'file_downloads', 'risk_score', 'anomaly_score']]
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title='3D Visualization of Behavior Patterns',
            height=700,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.9)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly Details Table
    st.subheader("Detailed Anomaly Analysis")
    
    anomaly_data = data[data['anomaly'] == -1].copy()
    anomaly_data['Severity'] = pd.qcut(anomaly_data['anomaly_score'], 
                                     q=3, 
                                     labels=['Low', 'Medium', 'High'])
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_severity = st.multiselect(
            "Filter by Severity",
            options=['Low', 'Medium', 'High'],
            default=['High']
        )
    
    with col2:
        selected_dept = st.multiselect(
            "Filter by Department",
            options=anomaly_data['department'].unique()
        )
    
    with col3:
        min_risk = st.number_input("Minimum Risk Score", 0, 100, 70)
    
    # Apply filters
    filtered_anomalies = anomaly_data[
        (anomaly_data['Severity'].isin(selected_severity) if selected_severity else True) &
        (anomaly_data['department'].isin(selected_dept) if selected_dept else True) &
        (anomaly_data['risk_score'] >= min_risk)
    ]
    
    # Display filtered anomalies
    if not filtered_anomalies.empty:
        st.dataframe(
            filtered_anomalies[[
                'employee_id', 'department', 'risk_score', 
                'anomaly_score', 'Severity', 'location'
            ]].sort_values('anomaly_score', ascending=False),
            hide_index=True
        )
    else:
        st.info("No anomalies match the selected filters.")

elif selected_page == "Trend Analysis":
    st.markdown("### Trend Analysis")
    
    # Time range selector with presets
    preset = st.radio(
        "Select Time Range",
        ["Last 7 days", "Last 30 days", "Custom"],
        horizontal=True,
        label_visibility="visible"
    )
    
    if preset == "Last 7 days":
        end_date = data['timestamp'].max()
        start_date = end_date - timedelta(days=7)
    elif preset == "Last 30 days":
        end_date = data['timestamp'].max()
        start_date = end_date - timedelta(days=30)
    else:
        date_range = st.date_input(
            "Select Custom Date Range",
            value=(data['timestamp'].min(), data['timestamp'].max())
        )
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])

    filtered_data = data[
        (data['timestamp'].dt.date >= start_date.date()) &
        (data['timestamp'].dt.date <= end_date.date())
    ]

    # Enhanced trend visualization
    st.subheader("Activity Trends")
    
    # Multi-metric selector with descriptions
    metrics_info = {
        'access_frequency': 'Number of system accesses',
        'email_volume': 'Number of emails sent/received',
        'file_downloads': 'Number of files downloaded',
        'risk_score': 'Calculated risk score'
    }
    
    selected_metrics = st.multiselect(
        "Select metrics to display",
        list(metrics_info.keys()),
        default=['access_frequency', 'risk_score'],
        format_func=lambda x: f"{x.replace('_', ' ').title()} ({metrics_info[x]})"
    )

    if selected_metrics:
        fig = go.Figure()
        colors = ['#4A90E2', '#2ECC71', '#E74C3C', '#F1C40F']
        
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Scatter(
                x=filtered_data['timestamp'],
                y=filtered_data[metric],
                name=metric.replace('_', ' ').title(),
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=f"{metric.replace('_', ' ').title()}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>"
            ))
        
        fig.update_layout(
            title_text="Trend Analysis",
            title_x=0.5,
            title_font=dict(size=16, family="Arial, sans-serif"),
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", color='#2d3748'),
            margin=dict(t=50, b=50, l=50, r=20),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(size=14),
                tickfont=dict(size=12),
                title="Date"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(size=14),
                tickfont=dict(size=12),
                title="Value"
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            )
        )
        
        # Add trend statistics
        st.subheader("Trend Statistics")
        cols = st.columns(len(selected_metrics))
        for i, metric in enumerate(selected_metrics):
            with cols[i]:
                current_value = filtered_data[metric].iloc[-1]
                previous_value = filtered_data[metric].iloc[-2]
                change = ((current_value - previous_value) / previous_value) * 100
                
                st.metric(
                    metric.replace('_', ' ').title(),
                    f"{current_value:.2f}",
                    f"{change:+.2f}%"
                )

elif selected_page == "Advanced Analytics":
    st.markdown("### Advanced Analytics")
    
    # 3D PCA visualization
    st.subheader("3D Principal Component Analysis")
    
    fig = px.scatter_3d(
        data,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color='risk_score',
        color_continuous_scale='Viridis',
        size='risk_score',
        opacity=0.7
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            zaxis_title='Third Principal Component'
        ),
        title_text="3D PCA Visualization of Employee Behavior",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance analysis
    st.subheader("Feature Importance Analysis")
    
    features = ['access_frequency', 'email_volume', 'file_downloads', 
                'after_hours_access', 'usb_usage', 'print_jobs']
    
    importance = np.array([0.2, 0.15, 0.3, 0.15, 0.1, 0.1])  # Example importance scores
    
    fig = go.Figure([go.Bar(
        x=features,
        y=importance,
        marker_color=['#4A90E2', '#2ECC71', '#E74C3C', '#F1C40F', '#9B59B6', '#34495E'],
        text=importance.round(3),
        textposition='auto',
    )])
    
    fig.update_layout(
        title_text="Feature Importance in Risk Assessment",
        title_x=0.5,
        title_font=dict(size=16, family="Arial, sans-serif"),
        xaxis_title="Features",
        yaxis_title="Importance Score",
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color='#2d3748'),
        margin=dict(t=50, b=50, l=50, r=20),
        xaxis=dict(
            tickangle=-45,
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, max(importance) * 1.1]
        ),
        bargap=0.3,
        showlegend=False
    )

# Footer
st.markdown("""
    <div class='dashboard-footer'>
        <p>Enterprise Insider Threat Detection System | Version 2.0</p>
        <p>Last updated: {}</p>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
