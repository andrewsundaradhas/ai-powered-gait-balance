"""
Standardized protocol for collecting gait/balance data (Step 1)
"""

DATA_COLLECTION_PROTOCOL = {
	"Setup": {
		"Camera": "Minimum 60 FPS, 4K preferred",
		"Distance": "3-4 meters from subject",
		"Angle": "Sagittal plane (side view)",
		"Lighting": "Well-lit, no shadows",
		"Background": "Uniform, preferably white",
		"Markers": "Optional reflective markers at joints",
	},
	"Gait Tests": {
		"10-Meter Walk Test": {
			"protocol": "Walk at comfortable speed for 10 meters",
			"duration": "30 seconds",
			"repetitions": 3,
			"rest": "2 minutes between trials",
			"measurements": "speed, cadence, step length",
		},
		"Timed Up and Go": {
			"protocol": "Rise from chair, walk 3m, return, sit",
			"duration": "1 minute",
			"repetitions": 3,
			"measurements": "time, balance, fall risk",
		},
		"Tandem Walking": {
			"protocol": "Walk in straight line, heel-to-toe",
			"duration": "20 seconds",
			"repetitions": 2,
			"measurements": "balance, coordination",
		},
		"Gait with Turns": {
			"protocol": "180° turns at end of walkway",
			"duration": "1 minute",
			"repetitions": 2,
			"measurements": "turning strategy, stability",
		},
	},
	"Balance Tests": {
		"Static Standing": {
			"protocol": "Stand still, eyes open, feet together",
			"duration": "60 seconds",
			"repetitions": 2,
			"measurements": "postural sway, CoM variation",
		},
		"Single Leg Stance": {
			"protocol": "Stand on one leg, other knee flexed",
			"duration": "30 seconds per leg",
			"measurements": "balance duration, hip control",
		},
		"Narrow Base Stance": {
			"protocol": "Stand with feet together, semi-tandem",
			"duration": "30 seconds",
			"measurements": "stability, tremor",
		},
		"Dynamic Reach": {
			"protocol": "Reach forward without stepping",
			"duration": "3 reaches per side",
			"measurements": "reach distance, stability",
		},
	},
	"Subject Info to Collect": [
		"Age, Gender, Height, Weight",
		"Medical History, Current Medications",
		"Previous Injuries, Surgeries",
		"Neurological Diagnosis (if any)",
		"Falls in past 12 months",
		"Balance confidence rating (1-10)",
		"Pain levels (if any)",
		"Consent forms",
	],
}

# Minimum subjects recommended:
# - 50 healthy controls (ages 20-80)
# - 30 Parkinson's disease
# - 20 Cerebellar ataxia
# - 30 Stroke patients
# - 20 Peripheral neuropathy
# - 20 Normal aging


