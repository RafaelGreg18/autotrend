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

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="AutoTrend Controller", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    init_db()

###############
#   Routes    #
###############

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


@app.post("/tickers")
async def create_ticker(ticker_name: str, db: Session = Depends(get_db)):
    """Create a new ticker"""
    if db.query(Ticker).filter(Ticker.ticker_name == ticker_name).first() is not None:
        raise HTTPException(status_code=400, detail="Ticker already exists")
    
    new_ticker = Ticker(ticker_name=ticker_name, last_update=date.today(), is_active=True)
    db.add(new_ticker)
    db.commit()
    db.refresh(new_ticker)
    
    return {
        "message": f"Ticker {ticker_name} created successfully",
        "ticker": {
            "ticker_name": new_ticker.ticker_name,
            "last_update": new_ticker.last_update,
            "is_active": new_ticker.is_active
        }
    }


@app.put("/tickers/{ticker_name}/update-date")
async def update_ticker_date(ticker_name: str, db: Session = Depends(get_db)):
    """Update ticker's last_update to today"""
    ticker = db.query(Ticker).filter(Ticker.ticker_name == ticker_name).first()
    if ticker is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    ticker.last_update = date.today()
    db.commit()
    
    return {
        "message": f"Ticker {ticker_name} date updated successfully",
        "last_update": ticker.last_update
    }

@app.put("/tickers/{ticker_name}/update-status")
async def update_ticker_status(ticker_name: str, is_active: bool, db: Session = Depends(get_db)):
    """Update ticker's active status"""
    ticker = db.query(Ticker).filter(Ticker.ticker_name == ticker_name).first()
    if ticker is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    ticker.is_active = is_active
    db.commit()
    
    return {
        "message": f"Ticker {ticker_name} status updated successfully",
        "is_active": ticker.is_active
    }

@app.get("/active-tickers")
async def get_active_tickers(db: Session = Depends(get_db)):
    """Get all active tickers"""
    active_tickers = db.query(Ticker).filter(Ticker.is_active == True).all()
    return {"active_tickers": [
        {
            "ticker_name": ticker.ticker_name,
            "last_update": ticker.last_update
        } for ticker in active_tickers
    ]}

# delete all tickers
@app.delete("/tickers")
async def delete_all_tickers(db: Session = Depends(get_db)):
    """Delete all tickers from database"""
    db.query(Ticker).delete()
    db.commit()
    return {"message": "All tickers deleted successfully"}

@app.delete("/tickers/{ticker_name}")
async def delete_ticker(ticker_name: str, db: Session = Depends(get_db)):
    """Delete a specific ticker"""
    ticker = db.query(Ticker).filter(Ticker.ticker_name == ticker_name).first()
    if ticker is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    db.delete(ticker)
    db.commit()
    
    return {"message": f"Ticker {ticker_name} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)