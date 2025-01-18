from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class VQAHistory(Base):
    __tablename__ = 'vqa_history'
    
    id = Column(Integer, primary_key=True)
    image_path = Column(String)
    question = Column(String)
    predicted_answer = Column(String)
    actual_answer = Column(String)
    is_correct = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.now)

# Tạo engine và session
engine = create_engine('sqlite:///vqa_history.db')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
