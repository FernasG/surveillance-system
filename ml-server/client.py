import grpc
import ml_server_pb2
import ml_server_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = ml_server_pb2_grpc.MLServerStub(channel)

        print("--- Testing Embeddings ---")
        emb_request = ml_server_pb2.EncodeTextRequest(text="gRPC is incredibly fast.")
        emb_response = stub.EncodeText(emb_request)
        
        print(f"Sample vector data (first 5 dimensions): {emb_response.embedding[:5]}")

if __name__ == '__main__':
    run()