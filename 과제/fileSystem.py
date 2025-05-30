# FastAPI 기초 198p
# • 단일/다중 파일 업로드
# • 파일 목록 조회 : 페이징, 필터링 지원
# • 파일 다운로드 : 단일/다중 ZIP 다운로드
# • 파일 삭제 : 단일/다중 삭제
import logging
import os
import io
import zipfile
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from starlette.status import HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST

# 상수 정의
UPLOAD_DIRECTORY = "uploads"
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1GB
LOG_DIRECTORY = "./log"
LOG_FILENAME = "fileSystem_error.log"

# FastAPI 앱 초기화
app = FastAPI(
	title="File System API",
	description="File System API",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc",
	contact={
		"name": "Kim Jae Hyun",
		"phone": "010-7612-8629",
		"address": "Guri, South Korea",
		"email": "Kimjeahyun77@gmail.com",
		"url": "https://github.com/Novelike",
	},
)

# 로깅 설정
os.makedirs(LOG_DIRECTORY, exist_ok=True)
logging.basicConfig(
	filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
	level=logging.ERROR,
	encoding='utf-8',
)
logger = logging.getLogger(__name__)


def ensure_upload_directory_exists():
	"""업로드 디렉토리가 존재하는지 확인하고, 없으면 생성합니다."""
	os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


def get_file_info(file_name: str) -> Dict[str, Any]:
	"""파일에 대한 메타데이터를 반환합니다."""
	file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
	return {
		"name": file_name,
		"size": os.path.getsize(file_path),
		"modified_at": os.path.getmtime(file_path)
	}


def validate_file_uploads(files: List[UploadFile]) -> None:
	"""파일 업로드 유효성 검사를 수행합니다."""
	# 파일 이름으로 중복 검사
	file_names = [file.filename for file in files]
	if len(file_names) != len(set(file_names)):
		logger.error("중복 파일 업로드 감지됨")
		raise HTTPException(status_code=400, detail="Duplicate file upload detected.")


def validate_files_exist(filenames: List[str]) -> None:
	"""요청된 모든 파일이 존재하는지 확인합니다."""
	ensure_upload_directory_exists()

	for filename in filenames:
		file_path = os.path.join(UPLOAD_DIRECTORY, filename)
		if not os.path.exists(file_path) or not os.path.isfile(file_path):
			logger.error(f"파일을 찾을 수 없음: {filename}")
			raise HTTPException(
				status_code=HTTP_404_NOT_FOUND,
				detail=f"File not found: {filename}"
			)


@app.get("/")
async def root():
	"""API 루트 엔드포인트"""
	return {"message": "Welcome to the KJH's File System API!"}


@app.post("/upload", response_model=Dict[str, Any])
async def file_uploads(files: list[UploadFile] = File(...)):
	"""
	# 여러 파일의 업로드를 처리합니다.
	## 중복된 파일이 업로드되지 않도록 하고 전체 파일 크기를 1GB로 제한하며, 지정된 디렉토리에 파일들을 안전하게 저장합니다.
	* ### **param files**: 업로드할 파일 객체 목록
	* ### **type files**: list[UploadFile]
	* ### **return**: 성공 메시지, 저장된 파일명 목록 및 각각의 파일 크기(바이트)가 포함된 딕셔너리
	* ### **rtype**: Dict[str, Any]
	"""
	ensure_upload_directory_exists()
	validate_file_uploads(files)

	total_size = 0
	saved_files = []
	file_sizes = []

	try:
		for file in files:
			# 안전한 파일 이름 생성
			safe_filename = os.path.basename(file.filename)
			file_path = os.path.join(UPLOAD_DIRECTORY, safe_filename)

			# 파일 내용 한 번만 읽기
			content = await file.read()
			file_size = len(content)
			file_sizes.append(file_size)
			total_size += file_size

			# 총 파일 크기 확인
			if total_size > MAX_UPLOAD_SIZE:
				logger.error("파일 크기가 제한을 초과함")
				raise HTTPException(status_code=400, detail="File size exceeds the limit.")

			# 파일 저장
			with open(file_path, "wb") as buffer:
				buffer.write(content)
			saved_files.append(safe_filename)

		return {
			"message": "File upload successful.",
			"file_names": saved_files,
			"file_sizes": file_sizes,
		}
	except Exception as e:
		logger.error(f"파일 업로드 중 오류: {str(e)}")
		raise HTTPException(status_code=500, detail="File upload failed")


@app.get("/files", response_model=Dict[str, Any])
async def get_paginated_files(
		page: int = 1,
		page_size: int = 10,
		search: str = ""
) -> Dict[str, Any]:
	"""
	# 페이지네이션과 검색 기능을 지원하는 업로드된 파일 목록을 반환합니다.
	### Parameters
	- **page**: 요청할 페이지 번호 (기본값: 1)
	- **page_size**: 페이지당 파일 수 (기본값: 10)
	- **search**: 파일 이름 검색 문자열 (기본값: "")
	### Returns
	**Dictionary** containing:
	- **total**: 전체 파일 수
	- **page**: 현재 페이지 번호
	- **page_size**: 페이지당 파일 수
	- **total_pages**: 전체 페이지 수
	- **files**: 파일 목록
	"""
	ensure_upload_directory_exists()

	# 모든 파일 목록 가져오기
	all_files = [f for f in os.listdir(UPLOAD_DIRECTORY) if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, f))]

	# 검색 필터 적용
	if search:
		all_files = [f for f in all_files if search.lower() in f.lower()]

	# 페이지네이션 적용
	total_files = len(all_files)
	total_pages = (total_files + page_size - 1) // page_size
	start_idx = (page - 1) * page_size
	end_idx = min(start_idx + page_size, total_files)
	paginated_files = all_files[start_idx:end_idx]

	# 파일 정보 구성
	file_list = [get_file_info(file_name) for file_name in paginated_files]

	return {
		"total": total_files,
		"page": page,
		"page_size": page_size,
		"total_pages": total_pages,
		"files": file_list
	}


# 파일 목록에서 파일을 선택하여 다운로드함(Zip파일로 압축)
@app.get("/download")
async def file_downloads(filenames: List[str] = Query(...)):
	"""
	# 파일 다운로드 API
	## 선택한 파일을 다운로드합니다. 여러 파일을 선택한 경우 ZIP 파일로 압축하여 다운로드합니다.

	### Parameters
	- **filenames**: 다운로드할 파일 이름 목록

	### Returns
	- 단일 파일: 원본 파일 다운로드
	- 여러 파일: ZIP 압축 파일 다운로드
	"""
	# 파일 존재 확인
	validate_files_exist(filenames)

	# 단일 파일 다운로드
	if len(filenames) == 1:
		filename = filenames[0]
		file_path = os.path.join(UPLOAD_DIRECTORY, filename)
		return FileResponse(
			path=file_path,
			filename=filename,
			media_type="application/octet-stream"
		)

	# 여러 파일 ZIP 압축 다운로드
	else:
		# 메모리 내 ZIP 파일 생성
		zip_io = io.BytesIO()
		with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
			for filename in filenames:
				file_path = os.path.join(UPLOAD_DIRECTORY, filename)
				# 파일을 ZIP에 추가
				zip_file.write(file_path, arcname=filename)

		# 스트림 포인터를 처음으로 이동
		zip_io.seek(0)

		# ZIP 파일을 클라이언트에게 스트리밍
		return StreamingResponse(
			zip_io,
			media_type="application/zip",
			headers={
				"Content-Disposition": f"attachment; filename=files.zip"
			}
		)

