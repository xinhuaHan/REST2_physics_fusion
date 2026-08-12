(PyTorch-2.1.0) [ma-user REST2_physics_fusion-why]$python scripts/inspect_ylj_parquet.py   --parquet /data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet   --pwat-unit mm   --output-json outputs/ylj_parquet_inspection.json
{
  "path": "/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet",
  "expected_step_minutes": 15,
  "declared_pwat_unit": "mm",
  "declared_pwat_to_cm": 0.1,
  "problems": [
    "missing expected columns: ['GHI_observe', 'DNI_observe', 'DHI_observe', 'TEMP_observe', 'WS_observe', 'WD_observe', 'PREC_observe', 'PWAT_observe', 'SDWE_observe', 'GHI_forecast_1day', 'TEMP_forecast_1day', 'WS_forecast_1day', 'WD_forecast_1day', 'PREC_forecast_1day', 'PWAT_forecast_1day', 'SDWE_forecast_1day', 'GHI_forecast_4hour', 'TEMP_forecast_4hour', 'WS_forecast_4hour', 'WD_forecast_4hour', 'PREC_forecast_4hour', 'PWAT_forecast_4hour', 'SDWE_forecast_4hour']",
    "PWAT values do not support the declared mm unit: inference=unknown"
  ],
  "file": {
    "size_bytes": 1897970,
    "rows": 49632,
    "row_groups": 1,
    "created_by": "parquet-cpp-arrow version 21.0.0"
  },
  "schema": {
    "columns": [
      "timestamp",
      "observe_power",
      "observe_ghi-onsite",
      "estimated_DNI-onsite",
      "estimated_DHI-onsite",
      "GHI-NWP_observe",
      "TEMP-NWP_observe",
      "WS-NWP_observe",
      "WD-NWP_observe",
      "PREC-NWP_observe",
      "PWAT-NWP_observe",
      "SDWE-NWP_observe",
      "GHI-NWP_forecast_1day",
      "TEMP-NWP_forecast_1day",
      "WS-NWP_forecast_1day",
      "WD-NWP_forecast_1day",
      "PREC-NWP_forecast_1day",
      "PWAT-NWP_forecast_1day",
      "SDWE-NWP_forecast_1day",
      "GHI-NWP_forecast_4hour",
      "TEMP-NWP_forecast_4hour",
      "WS-NWP_forecast_4hour",
      "WD-NWP_forecast_4hour",
      "PREC-NWP_forecast_4hour",
      "PWAT-NWP_forecast_4hour",
      "SDWE-NWP_forecast_4hour"
    ],
    "missing_expected_columns": [
      "GHI_observe",
      "DNI_observe",
      "DHI_observe",
      "TEMP_observe",
      "WS_observe",
      "WD_observe",
      "PREC_observe",
      "PWAT_observe",
      "SDWE_observe",
      "GHI_forecast_1day",
      "TEMP_forecast_1day",
      "WS_forecast_1day",
      "WD_forecast_1day",
      "PREC_forecast_1day",
      "PWAT_forecast_1day",
      "SDWE_forecast_1day",
      "GHI_forecast_4hour",
      "TEMP_forecast_4hour",
      "WS_forecast_4hour",
      "WD_forecast_4hour",
      "PREC_forecast_4hour",
      "PWAT_forecast_4hour",
      "SDWE_forecast_4hour"
    ],
    "unexpected_columns": [
      "observe_ghi-onsite",
      "estimated_DNI-onsite",
      "estimated_DHI-onsite",
      "GHI-NWP_observe",
      "TEMP-NWP_observe",
      "WS-NWP_observe",
      "WD-NWP_observe",
      "PREC-NWP_observe",
      "PWAT-NWP_observe",
      "SDWE-NWP_observe",
      "GHI-NWP_forecast_1day",
      "TEMP-NWP_forecast_1day",
      "WS-NWP_forecast_1day",
      "WD-NWP_forecast_1day",
      "PREC-NWP_forecast_1day",
      "PWAT-NWP_forecast_1day",
      "SDWE-NWP_forecast_1day",
      "GHI-NWP_forecast_4hour",
      "TEMP-NWP_forecast_4hour",
      "WS-NWP_forecast_4hour",
      "WD-NWP_forecast_4hour",
      "PREC-NWP_forecast_4hour",
      "PWAT-NWP_forecast_4hour",
      "SDWE-NWP_forecast_4hour"
    ],
    "arrow_types": {
      "timestamp": "timestamp[ns]",
      "observe_power": "double",
      "observe_ghi-onsite": "double",
      "estimated_DNI-onsite": "double",
      "estimated_DHI-onsite": "double",
      "GHI-NWP_observe": "double",
      "TEMP-NWP_observe": "double",
      "WS-NWP_observe": "double",
      "WD-NWP_observe": "double",
      "PREC-NWP_observe": "double",
      "PWAT-NWP_observe": "double",
      "SDWE-NWP_observe": "double",
      "GHI-NWP_forecast_1day": "double",
      "TEMP-NWP_forecast_1day": "double",
      "WS-NWP_forecast_1day": "double",
      "WD-NWP_forecast_1day": "double",
      "PREC-NWP_forecast_1day": "double",
      "PWAT-NWP_forecast_1day": "double",
      "SDWE-NWP_forecast_1day": "double",
      "GHI-NWP_forecast_4hour": "double",
      "TEMP-NWP_forecast_4hour": "double",
      "WS-NWP_forecast_4hour": "double",
      "WD-NWP_forecast_4hour": "double",
      "PREC-NWP_forecast_4hour": "double",
      "PWAT-NWP_forecast_4hour": "double",
      "SDWE-NWP_forecast_4hour": "double"
    }
  },
  "timestamps": {
    "parse_failures": 0,
    "duplicates": 0,
    "monotonic_in_file": true,
    "minimum": "2024-01-01T00:00:00",
    "maximum": "2025-05-31T23:45:00",
    "dominant_interval_minutes": 15.0,
    "interval_distribution_top10": [
      {
        "minutes": 15.0,
        "count": 49631
      }
    ]
  },
  "numeric": {
    "observe_power": {
      "rows": 49632,
      "finite": 42930,
      "missing": 6702,
      "parse_failures": 0,
      "minimum": -1.857,
      "q01": -1.589,
      "q05": -1.463,
      "median": 0.0,
      "mean": 84.06475266713252,
      "q95": 383.91775,
      "q99": 427.57917999999995,
      "maximum": 464.914,
      "negative_count": 19002,
      "zero_count": 5048
    }
  },
  "pwat_unit_check": {
    "inferred_unit": "unknown",
    "confidence": "none",
    "reason": "no finite PWAT values"
  },
  "sample": [
    {
      "timestamp": "2024-01-01T00:00:00.000",
      "observe_power": -0.964
    },
    {
      "timestamp": "2024-01-01T00:15:00.000",
      "observe_power": -0.979
    },
    {
      "timestamp": "2024-01-01T00:30:00.000",
      "observe_power": -1.122
    },
    {
      "timestamp": "2024-01-01T00:45:00.000",
      "observe_power": -1.277
    },
    {
      "timestamp": "2024-01-01T01:00:00.000",
      "observe_power": -1.281
    }
  ],
  "usable": false
