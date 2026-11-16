"""Shared semantic palette for final C-problem publication figures.

This module changes presentation only; it never changes plotted data or model
results. Warm colors encode cost/risk; cool colors encode energy states and
dispatch quantities.
"""

ACTUAL_NET_LOAD = "#4573B4"
SOC_MAIN = "#4573B4"
PLAN_PURCHASE = "#6090C1"
FORECAST_NET_LOAD = "#ACD2E5"
PRICE = "#F79015"
EMERGENCY_PURCHASE = "#D7312D"
SOC_LOWER_BOUND = "#E48070"
SOC_UPPER_BOUND = "#6B7280"
GRID_GREY = "#E8E8E8"
AXIS_GREY = "#3D3D3D"
FILL_ALPHA = 0.10

# Backward-compatible aliases for non-final supporting figures.
DEEP_BLUE = ACTUAL_NET_LOAD
PINK_PURPLE = EMERGENCY_PURCHASE
AMBER = PRICE
MINT = "#AEDBD2"
SLATE_PURPLE = SOC_UPPER_BOUND
LIGHT_LAVENDER = "#D5E9C6"
LIGHT_BLUE = FORECAST_NET_LOAD
