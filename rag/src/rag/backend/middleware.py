import time

import mlflow
from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

HTTP_CLIENT_ERROR_THRESHOLD = 400


def logging_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def mlflow_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start_time = time.perf_counter()

        with mlflow.start_run(run_name=f"{request.method} {request.url.path}"):
            response = await call_next(request)
            elapsed_time_seconds = time.perf_counter() - start_time

            mlflow.log_metric("status_code", response.status_code)
            mlflow.log_metric("latency_seconds", elapsed_time_seconds)

            mlflow.log_params(
                {
                    "endpoint": request.url.path,
                    "method": request.method,
                    "is_error": response.status_code >= HTTP_CLIENT_ERROR_THRESHOLD,
                },
            )

            mlflow.set_tags(
                {
                    "environment": "dev",
                    "client_ip": request.url.path,
                    "method": request.method,
                },
            )
        return response
