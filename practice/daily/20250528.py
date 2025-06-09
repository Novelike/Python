import random
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

app = FastAPI()


# 주문서 양식 정의
class Order(BaseModel):
	"""BaseModel : 데이터 유효성 검사, 타입 힌트, 자동 문서화 등을 지원하는 모델"""
	menu_item: str = Field(..., min_length=1, description="주문 메뉴")
	quantity: int = Field(..., gt=0, description="주문 수량")
	customer_name: str = Field(..., min_length=1, description="고객명")
	special_requests: Optional[str] = Field("없음", description="특별 요청사항")

	class Config:
		schema_extra = {
			"example": {
				"menu_item": "피자",
				"quantity": 1,
				"customer_name": "홍길동",
				"special_requests": "치즈 추가"
			}
		}


@app.get("/")
async def root():
	return {"message": "Welcome to the FastAPI Restaurant!"}


@app.post("/orders")
async def create_order(
		menu_item: str = Query(..., min_length=1),
		quantity: int = Query(..., gt=0),
		customer_name: str = Query(..., min_length=1),
		special_requests: Optional[str] = Query("없음")
):
	order = Order(
		menu_item=menu_item,
		quantity=quantity,
		customer_name=customer_name,
		special_requests=special_requests
	)
	return {
		"message": "주문이 접수되었습니다!",
		"order_details": {
			"메뉴": order.menu_item,
			"수량": order.quantity,
			"고객명": order.customer_name,
			"특별요청": order.special_requests
		},
		"order_id": random.randint(10000, 99999)
	}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
