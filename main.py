from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router, start_update_task

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает запросы с любых источников
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, PUT, DELETE и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки
)

app.include_router(router)

@app.on_event("startup")
async def on_startup():
    await start_update_task()  # Запуск задачи обновления токенов и моделей

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
