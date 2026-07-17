from collections import Counter
from loguru import logger
from guard.core.interfaces import AnalyticsStoreInterface

class AnalyticsService:
    def __init__(self, analytics_store: AnalyticsStoreInterface):
        self.store = analytics_store

    async def get_dashboard_metrics(self) -> dict:
        logger.info("Fetching overall analytics metrics")
        
        raw_data = await self.store.get_overall_dashboard_data()

        object_counter = Counter()
        for obj_string in raw_data["raw_objects_list"]:
            entities = [item.strip() for item in obj_string.split(",")]
            object_counter.update(entities)

        formatted_objects = [
            {"entity": entity, "count": count} 
            for entity, count in object_counter.most_common()
        ]

        return {
            "summary": {
                "total_events_overall": raw_data["total_events"]
            },
            "heatmap": raw_data["events_per_hour"],
            "object_distribution": formatted_objects
        }
