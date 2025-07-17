from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, String, Date, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import date
import os

from celery_controller import send_etl_task, send_train_task

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://autotrend_user:autotrend_pass@localhost:5432/autotrend")

# Create database engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
app = FastAPI(title="AutoTrend Controller", version="1.0.0")


# Database model
class Ticker(Base):
    __tablename__ = "tickers"
    
    ticker_name = Column(String(20), primary_key=True, index=True)
    last_update = Column(Date)
    is_active = Column(Boolean, default=True)

class Model(Base):
    __tablename__ = "models"
    
    ticker_name = Column(String(20), primary_key=True, index=True)
    last_update = Column(Date)
    f1_score = Column(Float, nullable=True)

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

@app.on_event("startup")
async def startup_event():
    init_db()

###############
#   Routes    #
###############

@app.post('/etl/trigger')
async def trigger_etl(ticker_name: str, start_date: str | None = None):
    """
    Trigger the ETL process for a given ticker (sends task to Celery).
    
    Args:
        ticker_name (str): The name of the ticker to download data for.
        start_date (str | None): The start date for downloading data. If None, will download historical data from the beginning.
    """
    send_etl_task(ticker_name, start_date)
    return {"message": f"ETL task triggered for {ticker_name} with start date {start_date}"}

@app.post('/train/trigger')
async def trigger_train(ticker_name: str,
                        input_size: int = 8,
                        hidden_size: int = 128,
                        num_layers: int = 3,
                        output_size: int = 2,
                        sequence_length: int = 30,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.2,
                        max_epochs: int = 45):
    """Trigger the training process for a given ticker (sends task to Celery)."""
    send_train_task(
        ticker_name=ticker_name,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        sequence_length=sequence_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_epochs=max_epochs,
    )
    # TODO: get answer from Celery task (f1 score), compare with previous model, save to database if better

## TODO: Inference API
# retrieve model from modelregistry
# run inference

# TODO: API to save model to modelregistry

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