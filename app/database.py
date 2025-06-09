from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# 데이터베이스 연결 URL
# 형식: mysql+pymysql://사용자명:비밀번호@호스트:포트번호/데이터베이스명
DATABASE_URL = "mysql+pymysql://root:1234@localhost:3306/kakao_test"

# 데이터베이스 엔진 생성
engine = create_engine(DATABASE_URL, echo=True)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모델 베이스 클래스
Base = declarative_base()

# 의존성: 데이터베이스 세션 획득
def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

# 연결 테스트
if __name__ == "__main__":
	try:
		# 1. 엔진 레벨에서 연결 테스트
		with engine.connect() as connection:
			# 2. 기본 쿼리 테스트 (테이블 불필요)
			result = connection.execute(text("SELECT VERSION as mysql_version"))
			version = result.fetchone()[0]
			print(f"MySQL Server version: {version}")

			# 3. ORM 세션 테스트
			db = next(get_db())
			print(f"ORM session: {db}")

			# 4. 데이터베이스 존재 확인
			result = db.execute(text("SELECT DATABASE() as current_db"))
			current_db = result.fetchone()[0]
			print(f"Current database: {current_db}")

	except Exception as e:
		print(f"Error: {e}")

	finally:
		if 'db' in locals():
			db.close()
			print("Session closed.")