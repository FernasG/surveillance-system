import asyncio
from datetime import date
from typing import Optional
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

    def update_event_analytics(self, event_id: str, objects_detected: str) -> None:
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE event_analytics
                    SET objects_detected = ?
                    WHERE event_id = ?
                    """,
                    (objects_detected, event_id)
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to update analytics for event {event_id}: {e}")

    async def get_dashboard_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> dict:
        return await asyncio.to_thread(self._get_dashboard_data_sync, start_date, end_date)

    def _get_dashboard_data_sync(self, start_date: Optional[date], end_date: Optional[date]) -> dict:
        data = {
            "total_events": 0,
            "events_per_hour": [],
            "raw_objects_list": []
        }

        conditions = []
        params = []

        if start_date is not None:
            conditions.append("date(timestamp) >= ?")
            params.append(start_date.isoformat())

        if end_date is not None:
            conditions.append("date(timestamp) <= ?")
            params.append(end_date.isoformat())

        date_filter = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        objects_conditions = conditions + ["objects_detected IS NOT NULL", "objects_detected != ''"]
        objects_filter = f"WHERE {' AND '.join(objects_conditions)}"

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(f"""
                    SELECT COUNT(id) as total
                    FROM event_analytics
                    {date_filter}
                """, params)
                data["total_events"] = cursor.fetchone()["total"]

                cursor.execute(f"""
                    SELECT strftime('%H', datetime(timestamp, 'localtime')) as hour,
                           COUNT(id) as count
                    FROM event_analytics
                    {date_filter}
                    GROUP BY hour
                    ORDER BY hour ASC
                """, params)
                data["events_per_hour"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(f"""
                    SELECT objects_detected
                    FROM event_analytics
                    {objects_filter}
                """, params)
                data["raw_objects_list"] = [row["objects_detected"] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching analytics data: {e}")

        return data