from fastapi import APIRouter, Depends, Request
from guard.api.routers.auth_router import get_current_user_token_data
from guard.core.services.analytics_service import AnalyticsService

router = APIRouter(tags=["Analytics"])

def get_analytics_service(request: Request) -> AnalyticsService:
    return request.state.analytics_service

@router.get("/metrics/daily", dependencies=[Depends(get_current_user_token_data)])
async def get_daily_metrics(analytics_service: AnalyticsService = Depends(get_analytics_service)):
    return await analytics_service.get_dashboard_metrics()

@router.get("/metrics/overall", dependencies=[Depends(get_current_user_token_data)])
async def get_overall_metrics(analytics_service: AnalyticsService = Depends(get_analytics_service)):
    return await analytics_service.get_dashboard_metrics()