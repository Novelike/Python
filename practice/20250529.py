from fastapi import FastAPI

app = FastAPI(
	title="KJH API",
	description="API for KJH",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc",
	contact={
		"name": "Kim Jae Hyun",
		"phone": "010-1234-5678",
		"address": "Seoul, South Korea",
		"email": "kjh@example.com",
		"url": "https://github.com/Novelike",
	},
)


@app.get("/books", tags=["도서 관리"])
def get_books():
	return "도서 관리"


@app.post("/books", tags=["도서 관리"])
def create_books():
	return "book 생성"


@app.get("/users", tags=["사용자 관리"])
def get_users():
	return "사용자 관리"


@app.post("/users", tags=["사용자 관리"])
def create_users():
	return "사용자 생성"
