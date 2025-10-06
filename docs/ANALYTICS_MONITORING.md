# CAIS Documentation Analytics & Monitoring

This document describes the analytics and monitoring system implemented for the CAIS documentation.

## Overview

The analytics and monitoring system provides:

1. **Real-time Analytics** - Track user behavior, page views, and engagement
2. **Feedback Collection** - Gather user feedback for continuous improvement
3. **Automated Monitoring** - Check for broken links, build failures, and performance issues
4. **Performance Metrics** - Monitor page load times and user experience
5. **Alert System** - Notify administrators of critical issues

## Components

### 1. Google Analytics Integration

**File**: `docs/source/_templates/analytics.html`

Features:
- GA4 tracking with enhanced events
- Documentation-specific metrics
- User journey tracking
- Search query monitoring
- Code copy tracking
- External link click tracking

**Configuration**:
```python
# In conf.py
html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  # Replace with actual GA4 ID
    'analytics_anonymize_ip': True,
}

html_context = {
    'google_analytics_id': 'G-XXXXXXXXXX',
    'google_tag_manager_id': 'GTM-XXXXXXX',  # Optional
}
```

### 2. Feedback Collection System

**File**: `docs/source/_static/feedback.js`

Features:
- Floating feedback button
- Quick rating system (1-3 scale)
- Detailed issue reporting
- GitHub issue integration (planned)
- Local data storage
- User role classification

**Usage**:
- Users see a feedback button in the bottom-right corner
- Quick feedback: Simple rating with optional comment
- Detailed feedback: Full issue report with categorization

### 3. Analytics Dashboard

**File**: `docs/source/_static/analytics_dashboard.js`

Features:
- Real-time metrics display
- Popular pages tracking
- User journey visualization
- System alerts
- Data export functionality
- Admin-only access

**Access**:
- Available on localhost or with `?admin=true` parameter
- Click the 📊 button in the top-left corner
- Dashboard shows key metrics and alerts

### 4. Documentation Monitoring

**File**: `docs/monitor_documentation.py`

Features:
- Automated build checking
- Broken link detection
- Performance monitoring
- Email notifications
- Continuous monitoring mode
- Detailed reporting

**Usage**:
```bash
# Single check
python docs/run_monitoring.py

# Continuous monitoring
python docs/monitor_documentation.py --continuous

# Custom configuration
python docs/monitor_documentation.py --config custom_config.json
```

## Setup Instructions

### 1. Google Analytics Setup

1. Create a Google Analytics 4 property
2. Get your Measurement ID (G-XXXXXXXXXX)
3. Update `docs/source/conf.py`:
   ```python
   html_theme_options = {
       'analytics_id': 'YOUR_GA4_ID',
       'analytics_anonymize_ip': True,
   }
   ```

### 2. Email Notifications Setup

1. Edit `docs/monitor_config.json`:
   ```json
   {
     "email_notifications": true,
     "email_recipients": ["admin@example.com"],
     "smtp_server": "smtp.gmail.com",
     "smtp_username": "your-email@gmail.com",
     "smtp_password": "your-app-password"
   }
   ```

2. For Gmail, use an App Password instead of your regular password

### 3. Monitoring Dependencies

Install required packages:
```bash
pip install requests beautifulsoup4 smtplib
```

## Configuration

### Analytics Configuration

**File**: `docs/source/conf.py`

Key settings:
- `analytics_id`: Google Analytics Measurement ID
- `html_js_files`: JavaScript files to include
- `html_context`: Additional template variables

### Monitoring Configuration

**File**: `docs/monitor_config.json`

Key settings:
```json
{
  "base_url": "http://localhost:8000",
  "check_interval": 300,
  "max_response_time": 5.0,
  "email_notifications": false,
  "performance_thresholds": {
    "max_build_time": 120,
    "max_broken_links": 5,
    "min_performance_score": 80
  }
}
```

## Metrics Collected

### User Analytics
- Page views and unique visitors
- Session duration and bounce rate
- Search queries and results
- Navigation patterns
- Device and browser information
- Geographic location (anonymized)

### Performance Metrics
- Page load times
- Build times
- Server response times
- Error rates
- Broken link counts

### User Feedback
- Page helpfulness ratings
- Detailed issue reports
- Feature requests
- User role information

## Monitoring Alerts

### Alert Types

1. **Error Alerts**:
   - Build failures
   - Server downtime
   - Critical broken links
   - Performance degradation

2. **Warning Alerts**:
   - Build warnings
   - Minor broken links
   - Slow response times
   - Low feedback scores

### Alert Channels

1. **Email Notifications**:
   - Immediate alerts for critical issues
   - Daily summary reports
   - Weekly analytics reports

2. **Dashboard Alerts**:
   - Real-time status display
   - Historical alert tracking
   - Alert acknowledgment

## Data Privacy

### User Data Protection
- IP addresses are anonymized
- No personally identifiable information stored
- GDPR compliant data collection
- User consent mechanisms

### Data Retention
- Analytics data: 26 months (GA4 default)
- Feedback data: Stored locally, exported as needed
- Monitoring logs: 30 days retention

## Troubleshooting

### Common Issues

1. **Analytics not working**:
   - Check GA4 Measurement ID
   - Verify JavaScript files are loaded
   - Check browser console for errors

2. **Monitoring failures**:
   - Ensure dependencies are installed
   - Check network connectivity
   - Verify file permissions

3. **Email notifications not sent**:
   - Check SMTP configuration
   - Verify email credentials
   - Check spam folders

### Debug Mode

Enable debug logging:
```python
# In monitor_documentation.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Endpoints (Future)

Planned API endpoints for external integration:

- `GET /api/analytics/metrics` - Current metrics
- `GET /api/analytics/feedback` - Feedback data
- `POST /api/analytics/feedback` - Submit feedback
- `GET /api/monitoring/status` - System status
- `GET /api/monitoring/alerts` - Active alerts

## Contributing

To contribute to the analytics and monitoring system:

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure privacy compliance
5. Test with real data

## Support

For issues with analytics and monitoring:

1. Check the troubleshooting section
2. Review configuration files
3. Check system logs
4. Contact the development team

## License

This analytics and monitoring system is part of the CAIS project and follows the same license terms.