import os

def main():
    print("Hello from Docker!")
    print("ENV VAR:", os.getenv("APP_ENV"))

if __name__ == "__main__":
    main()