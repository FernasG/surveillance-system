import asyncio, json, logging
from redis import asyncio as aioredis
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService

logger = logging.getLogger(__name__)

class RedisQueueWorker:
    def __init__(self, redis_client: aioredis.Redis, inference_service: InferenceService, preprocessor_service: PreprocessorService):
        self.inference_service = inference_service
        self.preprocessor_service = preprocessor_service
        self._redis_client = redis_client

    async def start(self, queue_name: str):
        logger.info("Iniciando Worker do Redis (Assíncrono)...")
        
        try:
            while True:
                try:
                    result = await self._redis_client.brpop(queue_name)

                    if not result:
                        continue
                        
                    _, message = result
                    data = json.loads(message)
                    
                    logger.info(f"Processando novo vídeo: {data.get('video_path')}")
                    
                    frames = self.preprocessor_service.process(data["video_path"])
                    self.inference_service.inferer(frames)
                    
                except json.JSONDecodeError:
                    logger.error("Erro ao decodificar JSON da mensagem do Redis.")
                except Exception as e:
                    logger.error(f"Erro inesperado no pipeline do worker: {str(e)}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Worker do Redis interceptado pelo sinal de cancelamento.")
        finally:
            await self.stop()

    async def stop(self):
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Conexão com o Redis fechada com sucesso.")

        logger.info("Worker do Redis completamente parado.")