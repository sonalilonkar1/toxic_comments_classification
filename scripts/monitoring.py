"""Performance monitoring for toxic comment classification API."""

import time
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import threading
import logging
from collections import defaultdict, deque
import psutil
import os

class APIMonitor:
    """Monitor API performance and usage."""

    def __init__(self, log_dir: str = "logs/api_monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Setup logging
        self.setup_logging()

        # Monitoring data
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times

        # Model usage tracking
        self.model_usage = defaultdict(int)
        self.endpoint_usage = defaultdict(int)

        # Performance metrics
        self.start_time = time.time()
        self.peak_memory_usage = 0

        # Request history (last 24 hours)
        self.request_history = deque(maxlen=10000)

        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitoring_thread.start()

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / f"api_monitoring_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger('api_monitor')

    def log_request(self, endpoint: str, model: str, response_time: float,
                   status_code: int, error: Optional[str] = None):
        """Log an API request."""
        self.request_count += 1
        self.endpoint_usage[endpoint] += 1
        self.model_usage[model] += 1
        self.total_response_time += response_time
        self.response_times.append(response_time)

        if status_code >= 400:
            self.error_count += 1

        # Log to file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "model": model,
            "response_time": response_time,
            "status_code": status_code,
            "error": error
        }

        self.request_history.append(log_entry)
        self.logger.info(f"Request: {endpoint} - {model} - {response_time:.3f}s - Status: {status_code}")

    def get_stats(self) -> Dict:
        """Get current monitoring statistics."""
        current_time = time.time()
        uptime = current_time - self.start_time

        avg_response_time = self.total_response_time / self.request_count if self.request_count > 0 else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0

        # Calculate percentiles
        if self.response_times:
            response_times_list = list(self.response_times)
            response_times_list.sort()
            p50 = response_times_list[int(len(response_times_list) * 0.5)]
            p95 = response_times_list[int(len(response_times_list) * 0.95)]
            p99 = response_times_list[int(len(response_times_list) * 0.99)]
        else:
            p50 = p95 = p99 = 0

        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "p50_response_time": p50,
            "p95_response_time": p95,
            "p99_response_time": p99,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
            "model_usage": dict(self.model_usage),
            "endpoint_usage": dict(self.endpoint_usage),
            "peak_memory_mb": self.peak_memory_usage
        }

    def get_recent_requests(self, hours: int = 1) -> List[Dict]:
        """Get requests from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_requests = []

        for request in self.request_history:
            request_time = datetime.fromisoformat(request["timestamp"])
            if request_time > cutoff_time:
                recent_requests.append(request)

        return recent_requests

    def _background_monitor(self):
        """Background monitoring thread."""
        while True:
            try:
                # Monitor memory usage
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)

                # Save periodic stats
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self._save_periodic_stats()

                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)

    def _save_periodic_stats(self):
        """Save periodic statistics."""
        stats = self.get_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"stats_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

    def generate_report(self) -> Dict:
        """Generate a comprehensive monitoring report."""
        stats = self.get_stats()
        recent_requests = self.get_recent_requests(hours=1)

        # Analyze recent performance
        recent_response_times = [r["response_time"] for r in recent_requests]
        recent_errors = [r for r in recent_requests if r.get("status_code", 200) >= 400]

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary_stats": stats,
            "recent_activity": {
                "requests_last_hour": len(recent_requests),
                "errors_last_hour": len(recent_errors),
                "avg_response_time_last_hour": sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0
            },
            "model_performance": {},
            "recommendations": []
        }

        # Model-specific analysis
        for model, usage_count in self.model_usage.items():
            model_requests = [r for r in recent_requests if r["model"] == model]
            if model_requests:
                model_response_times = [r["response_time"] for r in model_requests]
                report["model_performance"][model] = {
                    "usage_count": usage_count,
                    "avg_response_time": sum(model_response_times) / len(model_response_times),
                    "error_rate": len([r for r in model_requests if r.get("status_code", 200) >= 400]) / len(model_requests)
                }

        # Generate recommendations
        if stats["error_rate"] > 0.05:
            report["recommendations"].append("High error rate detected. Check API logs for issues.")

        if stats["avg_response_time"] > 2.0:
            report["recommendations"].append("Average response time is high. Consider optimizing model inference.")

        if stats["p95_response_time"] > 5.0:
            report["recommendations"].append("95th percentile response time is high. Check for outliers.")

        return report

# Global monitor instance
api_monitor = APIMonitor()

def monitor_request(endpoint: str, model: str):
    """Decorator to monitor API requests."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            status_code = 200

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                status_code = 500
                raise
            finally:
                response_time = time.time() - start_time
                api_monitor.log_request(endpoint, model, response_time, status_code, error)

        return wrapper
    return decorator

def get_monitoring_stats():
    """Get current monitoring statistics."""
    return api_monitor.get_stats()

def get_monitoring_report():
    """Get comprehensive monitoring report."""
    return api_monitor.generate_report()

def save_monitoring_report():
    """Save monitoring report to file."""
    report = api_monitor.generate_report()
    report_dir = Path("reports/monitoring")
    report_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = report_dir / f"monitoring_report_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Monitoring report saved to: {filename}")
    return filename

# Example usage in Flask app
"""
from monitoring import api_monitor, monitor_request, get_monitoring_stats

@app.route('/health')
def health():
    return jsonify(get_monitoring_stats())

@app.route('/predict', methods=['POST'])
@monitor_request('predict', 'bert')  # Will use the model from request
def predict():
    # Your prediction logic here
    pass
"""