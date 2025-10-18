"""
SQLAlchemy ORM models scaffold (Step 8)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class User(Base):
	__tablename__ = "users"

	id = Column(Integer, primary_key=True)
	username = Column(String(100), unique=True)
	email = Column(String(100), unique=True)
	password_hash = Column(String(255))
	full_name = Column(String(100))
	date_of_birth = Column(DateTime)
	gender = Column(String(10))
	created_at = Column(DateTime, default=datetime.utcnow)

	patients = relationship("Patient", back_populates="user")
	sessions = relationship("AnalysisSession", back_populates="clinician")


class Patient(Base):
	__tablename__ = "patients"

	id = Column(Integer, primary_key=True)
	user_id = Column(Integer, ForeignKey("users.id"))
	height = Column(Float)
	weight = Column(Float)
	medical_history = Column(Text)
	medications = Column(Text)
	neurological_diagnosis = Column(String(100))
	falls_in_past_year = Column(Integer)
	balance_confidence_rating = Column(Integer)
	created_at = Column(DateTime, default=datetime.utcnow)

	user = relationship("User", back_populates="patients")
	sessions = relationship("AnalysisSession", back_populates="patient")


class AnalysisSession(Base):
	__tablename__ = "analysis_sessions"

	id = Column(Integer, primary_key=True)
	session_uuid = Column(String(36), unique=True)
	patient_id = Column(Integer, ForeignKey("patients.id"))
	clinician_id = Column(Integer, ForeignKey("users.id"))
	video_file_path = Column(String(255))
	video_duration_seconds = Column(Float)
	analysis_start_time = Column(DateTime)
	analysis_end_time = Column(DateTime)
	status = Column(String(50))
	error_message = Column(Text)
	created_at = Column(DateTime, default=datetime.utcnow)

	patient = relationship("Patient", back_populates="sessions")
	clinician = relationship("User", back_populates="sessions")
	gait_results = relationship("GaitAnalysisResult", back_populates="session")
	balance_results = relationship("BalanceAnalysisResult", back_populates="session")
	disorder_results = relationship("DisorderDetectionResult", back_populates="session")
	recommendations = relationship("ClinicalRecommendation", back_populates="session")
	features = relationship("AnalysisFeature", back_populates="session")


class GaitAnalysisResult(Base):
	__tablename__ = "gait_analysis_results"

	id = Column(Integer, primary_key=True)
	session_id = Column(Integer, ForeignKey("analysis_sessions.id"))
	gait_type = Column(String(100))
	gait_confidence = Column(Float)
	cadence_mean = Column(Float)
	cadence_std = Column(Float)
	stride_length_mean = Column(Float)
	stride_length_std = Column(Float)
	step_width_mean = Column(Float)
	step_width_std = Column(Float)
	knee_angle_symmetry = Column(Float)
	hip_angle_symmetry = Column(Float)
	ankle_angle_symmetry = Column(Float)
	arm_swing_amplitude_left = Column(Float)
	arm_swing_amplitude_right = Column(Float)
	posture_forward_lean = Column(Float)
	created_at = Column(DateTime, default=datetime.utcnow)

	session = relationship("AnalysisSession", back_populates="gait_results")


class BalanceAnalysisResult(Base):
	__tablename__ = "balance_analysis_results"

	id = Column(Integer, primary_key=True)
	session_id = Column(Integer, ForeignKey("analysis_sessions.id"))
	balance_status = Column(String(100))
	balance_confidence = Column(Float)
	com_sway_ml = Column(Float)
	com_sway_ap = Column(Float)
	com_sway_total = Column(Float)
	com_velocity_mean = Column(Float)
	com_velocity_max = Column(Float)
	com_velocity_std = Column(Float)
	postural_stability_index = Column(Float)
	fall_risk_level = Column(String(50))
	created_at = Column(DateTime, default=datetime.utcnow)

	session = relationship("AnalysisSession", back_populates="balance_results")


class DisorderDetectionResult(Base):
	__tablename__ = "disorder_detection_results"

	id = Column(Integer, primary_key=True)
	session_id = Column(Integer, ForeignKey("analysis_sessions.id"))
	suspected_disorder = Column(String(100))
	disorder_confidence = Column(Float)
	parkinsons_score = Column(Float)
	cerebellar_ataxia_score = Column(Float)
	peripheral_neuropathy_score = Column(Float)
	stroke_hemiplegia_score = Column(Float)
	severity_level = Column(String(50))
	created_at = Column(DateTime, default=datetime.utcnow)

	session = relationship("AnalysisSession", back_populates="disorder_results")


class ClinicalRecommendation(Base):
	__tablename__ = "clinical_recommendations"

	id = Column(Integer, primary_key=True)
	session_id = Column(Integer, ForeignKey("analysis_sessions.id"))
	recommendation_type = Column(String(100))
	priority = Column(String(20))
	recommendation_text = Column(Text)
	suggested_intervention = Column(String(255))
	follow_up_in_days = Column(Integer)
	created_at = Column(DateTime, default=datetime.utcnow)

	session = relationship("AnalysisSession", back_populates="recommendations")


class AnalysisFeature(Base):
	__tablename__ = "analysis_features"

	id = Column(Integer, primary_key=True)
	session_id = Column(Integer, ForeignKey("analysis_sessions.id"))
	feature_name = Column(String(100))
	feature_value = Column(Float)
	feature_type = Column(String(50))
	created_at = Column(DateTime, default=datetime.utcnow)

	session = relationship("AnalysisSession", back_populates="features")


