import asyncio
from loguru import logger

from guard.core.interfaces import AnalyticsStoreInterface
from guard.infrastructure.database.sqlite_store import SqliteDatabase

class SqliteAnalyticsStore(AnalyticsStoreInterface):
    def __init__(self, db: SqliteDatabase):
        self.db = db

    def save_event_analytics(self, event_id: str, objects_detected: str, timestamp: str) -> None:
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO event_analytics (event_id, objects_detected, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (event_id, objects_detected, timestamp)
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save analytics for event {event_id}: {e}")

    async def get_daily_dashboard_data(self) -> dict:
        return await asyncio.to_thread(self._get_daily_dashboard_data_sync)

    def _get_daily_dashboard_data_sync(self) -> dict:
        data = {
            "total_events": 0,
            "events_per_hour": [],
            "raw_objects_list": []
        }

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT COUNT(id) as total 
                    FROM event_analytics 
                    WHERE date(timestamp) = date('now', 'localtime')
                """)
                data["total_events"] = cursor.fetchone()["total"]

                cursor.execute("""
                    SELECT strftime('%H', datetime(timestamp, 'localtime')) as hour, 
                           COUNT(id) as count 
                    FROM event_analytics 
                    WHERE date(timestamp) = date('now', 'localtime')
                    GROUP BY hour
                    ORDER BY hour ASC
                """)
                data["events_per_hour"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute("""
                    SELECT objects_detected 
                    FROM event_analytics 
                    WHERE date(timestamp) = date('now', 'localtime')
                    AND objects_detected IS NOT NULL AND objects_detected != ''
                """)
                data["raw_objects_list"] = [row["objects_detected"] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching analytics data: {e}")

        return data
    
    async def get_overall_dashboard_data(self) -> dict:
        return await asyncio.to_thread(self._get_overall_dashboard_data_sync)

    def _get_overall_dashboard_data_sync(self) -> dict:
        data = {
            "total_events": 0,
            "events_per_hour": [],
            "raw_objects_list": []
        }

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT COUNT(id) as total 
                    FROM event_analytics 
                """)

                data["total_events"] = cursor.fetchone()["total"]

                cursor.execute("""
                    SELECT strftime('%H', datetime(timestamp, 'localtime')) as hour, 
                           COUNT(id) as count 
                    FROM event_analytics 
                    GROUP BY hour
                    ORDER BY hour ASC
                """)

                data["events_per_hour"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute("""
                    SELECT objects_detected 
                    FROM event_analytics 
                    WHERE objects_detected IS NOT NULL AND objects_detected != ''
                """)

                data["raw_objects_list"] = [row["objects_detected"] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching analytics data: {e}")

        return data