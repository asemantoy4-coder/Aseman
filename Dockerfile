FROM python:3.9-slim

WORKDIR /app

# نصب وابستگی‌های سیستم
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# کپی requirements و نصب
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد برنامه
COPY . .

# ایجاد دایرکتوری‌های لازم
RUN mkdir -p logs .cache data

# پورت
EXPOSE 8080

# اجرای برنامه
CMD ["python", "main.py"]
