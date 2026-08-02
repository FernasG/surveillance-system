from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from guard.api.routers.auth_router import get_current_user_token_data
from guard.core.services.analytics_service import AnalyticsService

router = APIRouter(tags=["Analytics"])

def get_analytics_service(request: Request) -> AnalyticsService:
    return request.state.analytics_service

@router.get("/metrics/daily", dependencies=[Depends(get_current_user_token_data)])
async def get_daily_metrics(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    end = end_date or date.today()
    start = start_date or (end if end_date else date.today())

    if end < start:
        raise HTTPException(status_code=400, detail="end_date must be on or after start_date")

    return await analytics_service.get_dashboard_metrics(start, end)

@router.get("/metrics/overall", dependencies=[Depends(get_current_user_token_data)])
async def get_overall_metrics(analytics_service: AnalyticsService = Depends(get_analytics_service)):
    return await analytics_service.get_dashboard_metrics()
