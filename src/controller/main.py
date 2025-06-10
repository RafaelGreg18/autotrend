from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, String, Date, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import date
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://autotrend_user:autotrend_pass@localhost:5432/autotrend")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class Ticker(Base):
    __tablename__ = "tickers"
    
    ticker_name = Column(String(20), primary_key=True, index=True)
    last_update = Column(Date)
    is_active = Column(Boolean, default=True)

# FastAPI app
app = FastAPI(title="AutoTrend Controller", version="1.0.0")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Routes
@app.get("/")
async def hello_world():
    """Simple hello world endpoint"""
    return {"message": "Hello World from AutoTrend Controller!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "autotrend-controller"}

@app.get("/tickers")
async def get_tickers(db: Session = Depends(get_db)):
    """Get all tickers from database"""
    tickers = db.query(Ticker).all()
    return {"tickers": [
        {
            "ticker_name": ticker.ticker_name,
            "last_update": ticker.last_update,
            "is_active": ticker.is_active
        } for ticker in tickers
    ]}

@app.get("/tickers/{ticker_name}")
async def get_ticker(ticker_name: str, db: Session = Depends(get_db)):
    """Get specific ticker information"""
    ticker = db.query(Ticker).filter(Ticker.ticker_name == ticker_name).first()
    if ticker is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    return {
        "ticker_name": ticker.ticker_name,
        "last_update": ticker.last_update,
        "is_active": ticker.is_active
    }

@app.post("/tickers/{ticker_name}/update")
async def update_ticker_date(ticker_name: str, db: Session = Depends(get_db)):
    """Update ticker's last_update to today"""
    ticker = db.query(Ticker).filter(Ticker.ticker_name == ticker_name).first()
    if ticker is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    ticker.last_update = date.today()
    db.commit()
    
    return {
        "message": f"Ticker {ticker_name} updated successfully",
        "last_update": ticker.last_update
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)