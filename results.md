(PyTorch-2.1.0) [ma-user why]$cd REST2_physics_fusion-why
(PyTorch-2.1.0) [ma-user REST2_physics_fusion-why]$python scripts/inspect_ylj_parquet.py \
  --parquet /data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet \
  --pwat-unit mm \
  --output-json outputs/ylj_parquet_inspection.json
{
  "path": "/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet",
  "expected_step_minutes": 15,
  "declared_pwat_unit": "mm",
  "declared_pwat_to_cm": 0.1,
  "problems": [
    "missing expected columns: ['observe_power', 'DNI_observe', 'DHI_observe']"
  ],
  "file": {
    "size_bytes": 1722373,
    "rows": 49632,
    "row_groups": 1,
    "created_by": "parquet-cpp-arrow version 21.0.0"
  },
  "schema": {
    "columns": [
      "timestamp",
      "observe_ghi",
      "estimated_DNI",
      "estimated_DHI",
      "GHI_observe",
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
    "missing_expected_columns": [
      "observe_power",
      "DNI_observe",
      "DHI_observe"
    ],
    "unexpected_columns": [
      "observe_ghi",
      "estimated_DNI",
      "estimated_DHI"
    ],
    "arrow_types": {
      "timestamp": "timestamp[ns]",
      "observe_ghi": "double",
      "estimated_DNI": "double",
      "estimated_DHI": "double",
      "GHI_observe": "double",
      "TEMP_observe": "double",
      "WS_observe": "double",
      "WD_observe": "double",
      "PREC_observe": "double",
      "PWAT_observe": "double",
      "SDWE_observe": "double",
      "GHI_forecast_1day": "double",
      "TEMP_forecast_1day": "double",
      "WS_forecast_1day": "double",
      "WD_forecast_1day": "double",
      "PREC_forecast_1day": "double",
      "PWAT_forecast_1day": "double",
      "SDWE_forecast_1day": "double",
      "GHI_forecast_4hour": "double",
      "TEMP_forecast_4hour": "double",
      "WS_forecast_4hour": "double",
      "WD_forecast_4hour": "double",
      "PREC_forecast_4hour": "double",
      "PWAT_forecast_4hour": "double",
      "SDWE_forecast_4hour": "double"
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
    "GHI_observe": {
      "rows": 49632,
      "finite": 48392,
      "missing": 1240,
      "parse_failures": 0,
      "minimum": 0.0,
      "q01": 0.0,
      "q05": 0.0,
      "median": 2.0,
      "mean": 198.4099231277897,
      "q95": 781.0,
      "q99": 979.0,
      "maximum": 1163.0,
      "negative_count": 0,
      "zero_count": 24073
    },
    "PWAT_observe": {
      "rows": 49632,
      "finite": 48392,
      "missing": 1240,
      "parse_failures": 0,
      "minimum": 0.2,
      "q01": 0.9,
      "q05": 1.6,
      "median": 5.3,
      "mean": 7.2427901306001,
      "q95": 17.8,
      "q99": 20.3,
      "maximum": 22.3,
      "negative_count": 0,
      "zero_count": 0
    },
    "PWAT_forecast_1day": {
      "rows": 49632,
      "finite": 46656,
      "missing": 2976,
      "parse_failures": 0,
      "minimum": 0.2000000029802322,
      "q01": 0.7,
      "q05": 1.5,
      "median": 5.199999809265137,
      "mean": 7.080924213437212,
      "q95": 17.299999237060547,
      "q99": 19.700000762939453,
      "maximum": 21.200000762939453,
      "negative_count": 0,
      "zero_count": 0
    },
    "PWAT_forecast_4hour": {
      "rows": 49632,
      "finite": 46656,
      "missing": 2976,
      "parse_failures": 0,
      "minimum": 0.2000000029802322,
      "q01": 0.7,
      "q05": 1.5,
      "median": 5.199999809265137,
      "mean": 7.080924213437212,
      "q95": 17.299999237060547,
      "q99": 19.700000762939453,
      "maximum": 21.200000762939453,
      "negative_count": 0,
      "zero_count": 0
    }
  },
  "pwat_unit_check": {
    "inferred_unit": "mm",
    "confidence": "high",
    "reason": "median is on a multi-millimetre scale and q99 is below 100 mm",
    "median_across_columns": 5.199999809265137,
    "maximum_q99_across_columns": 20.3,
    "mm_to_cm_factor": 0.1,
    "converted_median_cm_if_mm": 0.5199999809265137,
    "converted_maximum_q99_cm_if_mm": 2.0300000000000002
  },
  "sample": [
    {
      "timestamp": "2024-01-01T00:00:00.000",
      "GHI_observe": 0.0,
      "PWAT_observe": 2.0,
      "PWAT_forecast_1day": 1.8,
      "PWAT_forecast_4hour": 1.8
    },
    {
      "timestamp": "2024-01-01T00:15:00.000",
      "GHI_observe": 0.0,
      "PWAT_observe": 2.1,
      "PWAT_forecast_1day": 1.8,
      "PWAT_forecast_4hour": 1.8
    },
    {
      "timestamp": "2024-01-01T00:30:00.000",
      "GHI_observe": 0.0,
      "PWAT_observe": 2.1,
      "PWAT_forecast_1day": 1.8,
      "PWAT_forecast_4hour": 1.8
    },
    {
      "timestamp": "2024-01-01T00:45:00.000",
      "GHI_observe": 0.0,
      "PWAT_observe": 2.1,
      "PWAT_forecast_1day": 1.9,
      "PWAT_forecast_4hour": 1.9
    },
    {
      "timestamp": "2024-01-01T01:00:00.000",
      "GHI_observe": 0.0,
      "PWAT_observe": 2.2,
      "PWAT_forecast_1day": 1.9,
      "PWAT_forecast_4hour": 1.9
    }
  ],
  "usable": false
}
(PyTorch-2.1.0) [ma-use
