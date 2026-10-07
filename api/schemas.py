from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    recency_days: int = Field(..., ge=0, description="Days since last purchase")
    frequency: int = Field(..., ge=1, description="Number of distinct orders")
    monetary_total: float = Field(..., ge=0, description="Total amount spent")
    avg_order_value: float = Field(..., ge=0, description="Average order value")
    distinct_products: int = Field(..., ge=0, description="Number of distinct products bought")
    tenure_days: int = Field(..., ge=0, description="Days between first and last purchase")
    num_returns: int = Field(0, ge=0, description="Number of returned line items")
    return_rate: float = Field(0.0, ge=0, description="num_returns / frequency")
    country_group: str = Field("UK", description="'UK' or 'Other'")

    class Config:
        json_schema_extra = {
            "example": {
                "recency_days": 74,
                "frequency": 12,
                "monetary_total": 850.50,
                "avg_order_value": 70.88,
                "distinct_products": 25,
                "tenure_days": 300,
                "num_returns": 1,
                "return_rate": 0.083,
                "country_group": "UK",
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float


class ModelInfoResponse(BaseModel):
    model_type: str
    best_params: dict
    metrics: dict
    features: list[str]


class HealthResponse(BaseModel):
    status: str
