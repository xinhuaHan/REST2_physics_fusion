(PyTorch-2.1.0) [ma-user REST2_physics_fusion-why]$python scripts/inspect_luoyang_parquet.py \
  --parquet /data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1-with_DNI_DHI.parquet \
  --output-json outputs/luoyang_parquet_inspection.json
{
  "path": "/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1-with_DNI_DHI.parquet",
  "expected_step_minutes": 5,
  "rated_power": 48629.73,
  "problems": [
    "missing required columns: ['asi_path', 'asi_path_timestamps', 'GHI_mean_observe', 'msl_forecast', 't2m_forecast', 'u10_forecast', 'v10_forecast', 'u100_forecast', 'v100_forecast']",
    "GHI candidate GHI-onsite contains no finite values",
    "GHI candidate GHI-onsite-timestamps contains no finite values",
    "DNI candidate estimated_DNI-onsite contains no finite values",
    "DNI candidate estimated_DNI-onsite-timestamps contains no finite values",
    "DHI candidate estimated_DHI-onsite contains no finite values",
    "DHI candidate estimated_DHI-onsite-timestamps contains no finite values"
  ],
  "file": {
    "size_bytes": 28667787,
    "rows": 122976,
    "row_groups": 1,
    "created_by": "parquet-cpp-arrow version 21.0.0"
  },
  "schema": {
    "columns": [
      "timestamp",
      "final_power",
      "GHI-onsite",
      "GHI-onsite-timestamps",
      "estimated_DNI-onsite",
      "estimated_DNI-onsite-timestamps",
      "estimated_DHI-onsite",
      "estimated_DHI-onsite-timestamps",
      "asi_path-onsite",
      "asi_path-onsite-timestamps",
      "GHI_mean-NWP_observe",
      "msl-NWP_observe",
      "t2m-NWP_observe",
      "u10-NWP_observe",
      "v10-NWP_observe",
      "u100-NWP_observe",
      "v100-NWP_observe",
      "GHI_mean-NWP_forecast",
      "msl-NWP_forecast",
      "t2m-NWP_forecast",
      "u10-NWP_forecast",
      "v10-NWP_forecast",
      "u100-NWP_forecast",
      "v100-NWP_forecast"
    ],
    "missing_required_columns": [
      "asi_path",
      "asi_path_timestamps",
      "GHI_mean_observe",
      "msl_forecast",
      "t2m_forecast",
      "u10_forecast",
      "v10_forecast",
      "u100_forecast",
      "v100_forecast"
    ],
    "irradiance_candidates": {
      "ghi": [
        "GHI-onsite",
        "GHI-onsite-timestamps",
        "GHI_mean-NWP_observe",
        "GHI_mean-NWP_forecast"
      ],
      "dni": [
        "estimated_DNI-onsite",
        "estimated_DNI-onsite-timestamps"
      ],
      "dhi": [
        "estimated_DHI-onsite",
        "estimated_DHI-onsite-timestamps"
      ]
    },
    "arrow_types": {
      "timestamp": "timestamp[ns]",
      "final_power": "double",
      "GHI-onsite": "list<element: double>",
      "GHI-onsite-timestamps": "list<element: timestamp[us]>",
      "estimated_DNI-onsite": "list<element: double>",
      "estimated_DNI-onsite-timestamps": "list<element: timestamp[us]>",
      "estimated_DHI-onsite": "list<element: double>",
      "estimated_DHI-onsite-timestamps": "list<element: timestamp[us]>",
      "asi_path-onsite": "list<element: string>",
      "asi_path-onsite-timestamps": "list<element: timestamp[us]>",
      "GHI_mean-NWP_observe": "double",
      "msl-NWP_observe": "double",
      "t2m-NWP_observe": "double",
      "u10-NWP_observe": "double",
      "v10-NWP_observe": "double",
      "u100-NWP_observe": "double",
      "v100-NWP_observe": "double",
      "GHI_mean-NWP_forecast": "double",
      "msl-NWP_forecast": "double",
      "t2m-NWP_forecast": "double",
      "u10-NWP_forecast": "double",
      "v10-NWP_forecast": "double",
      "u100-NWP_forecast": "double",
      "v100-NWP_forecast": "double"
    }
  },
  "timestamps": {
    "parse_failures": 0,
    "duplicates": 0,
    "monotonic_in_file": true,
    "minimum": "2025-04-11T00:00:00",
    "maximum": "2026-06-11T23:55:00",
    "dominant_interval_minutes": 5.0,
    "interval_distribution_top10": [
      {
        "minutes": 5.0,
        "count": 122975
      }
    ]
  },
  "numeric": {
    "final_power": {
      "rows": 122976,
      "finite": 122976,
      "missing": 0,
      "parse_failures": 0,
      "minimum": 0.0,
      "q01": 0.0,
      "q05": 0.0,
      "median": 95.09285645,
      "mean": 6274.858471254146,
      "q95": 29451.5098225,
      "q99": 35067.942755000004,
      "maximum": 41667.33252,
      "negative_count": 0,
      "zero_count": 60940
    },
    "GHI-onsite": {
      "rows": 122976,
      "finite": 0,
      "missing": 0,
      "parse_failures": 122976,
      "minimum": null,
      "q01": null,
      "q05": null,
      "median": null,
      "mean": null,
      "q95": null,
      "q99": null,
      "maximum": null,
      "negative_count": 0,
      "zero_count": 0
    },
    "GHI-onsite-timestamps": {
      "rows": 122976,
      "finite": 0,
      "missing": 0,
      "parse_failures": 122976,
      "minimum": null,
      "q01": null,
      "q05": null,
      "median": null,
      "mean": null,
      "q95": null,
      "q99": null,
      "maximum": null,
      "negative_count": 0,
      "zero_count": 0
    },
    "GHI_mean-NWP_observe": {
      "rows": 122976,
      "finite": 101085,
      "missing": 21891,
      "parse_failures": 0,
      "minimum": 0.0,
      "q01": 0.0,
      "q05": 0.0,
      "median": 23.06797325,
      "mean": 186.94511341723327,
      "q95": 807.45418,
      "q99": 962.8585760000002,
      "maximum": 1062.0919,
      "negative_count": 0,
      "zero_count": 44076
    },
    "GHI_mean-NWP_forecast": {
      "rows": 122976,
      "finite": 116009,
      "missing": 6967,
      "parse_failures": 0,
      "minimum": 0.0,
      "q01": 0.0,
      "q05": 0.069804127,
      "median": 11.78458218,
      "mean": 197.06569470549297,
      "q95": 836.6510395599998,
      "q99": 964.0631363359997,
      "maximum": 1054.5479,
      "negative_count": 0,
      "zero_count": 1215
    },
    "estimated_DNI-onsite": {
      "rows": 122976,
      "finite": 0,
      "missing": 0,
      "parse_failures": 122976,
      "minimum": null,
      "q01": null,
      "q05": null,
      "median": null,
      "mean": null,
      "q95": null,
      "q99": null,
      "maximum": null,
      "negative_count": 0,
      "zero_count": 0
    },
    "estimated_DNI-onsite-timestamps": {
      "rows": 122976,
      "finite": 0,
      "missing": 0,
      "parse_failures": 122976,
      "minimum": null,
      "q01": null,
      "q05": null,
      "median": null,
      "mean": null,
      "q95": null,
      "q99": null,
      "maximum": null,
      "negative_count": 0,
      "zero_count": 0
    },
    "estimated_DHI-onsite": {
      "rows": 122976,
      "finite": 0,
      "missing": 0,
      "parse_failures": 122976,
      "minimum": null,
      "q01": null,
      "q05": null,
      "median": null,
      "mean": null,
      "q95": null,
      "q99": null,
      "maximum": null,
      "negative_count": 0,
      "zero_count": 0
    },
    "estimated_DHI-onsite-timestamps": {
      "rows": 122976,
      "finite": 0,
      "missing": 0,
      "parse_failures": 122976,
      "minimum": null,
      "q01": null,
      "q05": null,
      "median": null,
      "mean": null,
      "q95": null,
      "q99": null,
      "maximum": null,
      "negative_count": 0,
      "zero_count": 0
    }
  },
  "target_quality": {
    "negative_values": 0,
    "missing_values": 0
  },
  "sample": [
    {
      "timestamp": "2025-04-11T00:00:00.000",
      "final_power": 0.0,
      "GHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "GHI-onsite-timestamps": [
        "1970-01-21T04:32:09.360",
        "1970-01-21T04:32:09.420",
        "1970-01-21T04:32:09.480",
        "1970-01-21T04:32:09.540",
        "1970-01-21T04:32:09.600"
      ],
      "GHI_mean-NWP_observe": 0.0,
      "GHI_mean-NWP_forecast": 0.27853134,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:09.360",
        "1970-01-21T04:32:09.420",
        "1970-01-21T04:32:09.480",
        "1970-01-21T04:32:09.540",
        "1970-01-21T04:32:09.600"
      ],
      "estimated_DHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite-timestamps": [
        "1970-01-21T04:32:09.360",
        "1970-01-21T04:32:09.420",
        "1970-01-21T04:32:09.480",
        "1970-01-21T04:32:09.540",
        "1970-01-21T04:32:09.600"
      ]
    },
    {
      "timestamp": "2025-04-11T00:05:00.000",
      "final_power": 0.0,
      "GHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "GHI-onsite-timestamps": [
        "1970-01-21T04:32:09.660",
        "1970-01-21T04:32:09.720",
        "1970-01-21T04:32:09.780",
        "1970-01-21T04:32:09.840",
        "1970-01-21T04:32:09.900"
      ],
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.280485903,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:09.660",
        "1970-01-21T04:32:09.720",
        "1970-01-21T04:32:09.780",
        "1970-01-21T04:32:09.840",
        "1970-01-21T04:32:09.900"
      ],
      "estimated_DHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite-timestamps": [
        "1970-01-21T04:32:09.660",
        "1970-01-21T04:32:09.720",
        "1970-01-21T04:32:09.780",
        "1970-01-21T04:32:09.840",
        "1970-01-21T04:32:09.900"
      ]
    },
    {
      "timestamp": "2025-04-11T00:10:00.000",
      "final_power": 0.0,
      "GHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "GHI-onsite-timestamps": [
        "1970-01-21T04:32:09.960",
        "1970-01-21T04:32:10.020",
        "1970-01-21T04:32:10.080",
        "1970-01-21T04:32:10.140",
        "1970-01-21T04:32:10.200"
      ],
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.282440467,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:09.960",
        "1970-01-21T04:32:10.020",
        "1970-01-21T04:32:10.080",
        "1970-01-21T04:32:10.140",
        "1970-01-21T04:32:10.200"
      ],
      "estimated_DHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite-timestamps": [
        "1970-01-21T04:32:09.960",
        "1970-01-21T04:32:10.020",
        "1970-01-21T04:32:10.080",
        "1970-01-21T04:32:10.140",
        "1970-01-21T04:32:10.200"
      ]
    },
    {
      "timestamp": "2025-04-11T00:15:00.000",
      "final_power": 0.0,
      "GHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "GHI-onsite-timestamps": [
        "1970-01-21T04:32:10.260",
        "1970-01-21T04:32:10.320",
        "1970-01-21T04:32:10.380",
        "1970-01-21T04:32:10.440",
        "1970-01-21T04:32:10.500"
      ],
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.28439503,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:10.260",
        "1970-01-21T04:32:10.320",
        "1970-01-21T04:32:10.380",
        "1970-01-21T04:32:10.440",
        "1970-01-21T04:32:10.500"
      ],
      "estimated_DHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite-timestamps": [
        "1970-01-21T04:32:10.260",
        "1970-01-21T04:32:10.320",
        "1970-01-21T04:32:10.380",
        "1970-01-21T04:32:10.440",
        "1970-01-21T04:32:10.500"
      ]
    },
    {
      "timestamp": "2025-04-11T00:20:00.000",
      "final_power": 0.0,
      "GHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "GHI-onsite-timestamps": [
        "1970-01-21T04:32:10.560",
        "1970-01-21T04:32:10.620",
        "1970-01-21T04:32:10.680",
        "1970-01-21T04:32:10.740",
        "1970-01-21T04:32:10.800"
      ],
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.286349593,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:10.560",
        "1970-01-21T04:32:10.620",
        "1970-01-21T04:32:10.680",
        "1970-01-21T04:32:10.740",
        "1970-01-21T04:32:10.800"
      ],
      "estimated_DHI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite-timestamps": [
        "1970-01-21T04:32:10.560",
        "1970-01-21T04:32:10.620",
        "1970-01-21T04:32:10.680",
        "1970-01-21T04:32:10.740",
        "1970-01-21T04:32:10.800"
      ]
    }
  ],
  "usable": false
}
