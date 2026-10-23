BANDS = [f"A{i:02d}" for i in range(64)]

CLASS_NAMES = ["mangrove", "salt_marsh", "tidal_flat", "water", "seasonal_water", "built_up",
               "barren", "coastal_forest", "cropland", "herbaceous_wetland", "other_land"]
CLASS_IDS = list(range(1, len(CLASS_NAMES) + 1))
MANGROVE = 1
