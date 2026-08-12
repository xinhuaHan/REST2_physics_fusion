(PyTorch-2.1.0) [ma-user REST2_physics_fusion-why]$python scripts/inspect_luoyang_parquet.py   --parquet /data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1-with_DNI_DHI.parquet   --output-json outputs/luoyang_parquet_inspection.json
{
  "path": "/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1-with_DNI_DHI.parquet",
  "expected_step_minutes": 5,
  "rated_power": 48629.73,
  "problems": [],
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
    "missing_required_columns": [],
    "irradiance_candidates": {
      "ghi": [
        "GHI-onsite",
        "GHI_mean-NWP_observe",
        "GHI_mean-NWP_forecast"
      ],
      "dni": [
        "estimated_DNI-onsite"
      ],
      "dhi": [
        "estimated_DHI-onsite"
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
    "GHI-onsite": {
      "container": "list",
      "rows": 122976,
      "rows_with_nonempty_list": 122976,
      "rows_with_finite_value": 19511,
      "list_length_minimum": 5,
      "list_length_median": 5.0,
      "list_length_maximum": 5,
      "elements": 614880,
      "finite": 97276,
      "missing_or_unparseable_elements": 517604,
      "malformed_rows": 103465,
      "minimum": 0.0,
      "q01": 0.0,
      "q05": 0.0,
      "median": 30.0,
      "mean": 223.65051708540642,
      "q95": 895.0,
      "q99": 973.2,
      "maximum": 1199.2,
      "negative_count": 0,
      "zero_count": 38091
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
      "container": "list",
      "rows": 122976,
      "rows_with_nonempty_list": 122976,
      "rows_with_finite_value": 19511,
      "list_length_minimum": 5,
      "list_length_median": 5.0,
      "list_length_maximum": 5,
      "elements": 614880,
      "finite": 97276,
      "missing_or_unparseable_elements": 517604,
      "malformed_rows": 103465,
      "minimum": 0.0,
      "q01": 0.0,
      "q05": 0.0,
      "median": 0.0,
      "mean": 177.08924363943305,
      "q95": 807.3575074096645,
      "q99": 899.0756757849249,
      "maximum": 1037.5600887160779,
      "negative_count": 0,
      "zero_count": 54716
    },
    "estimated_DHI-onsite": {
      "container": "list",
      "rows": 122976,
      "rows_with_nonempty_list": 122976,
      "rows_with_finite_value": 19511,
      "list_length_minimum": 5,
      "list_length_median": 5.0,
      "list_length_maximum": 5,
      "elements": 614880,
      "finite": 97276,
      "missing_or_unparseable_elements": 517604,
      "malformed_rows": 103465,
      "minimum": 0.0,
      "q01": 0.0,
      "q05": 0.0,
      "median": 28.0,
      "mean": 97.80327184669515,
      "q95": 376.507768569276,
      "q99": 444.0433029919428,
      "maximum": 514.1000891053231,
      "negative_count": 0,
      "zero_count": 38091
    },
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
    }
  },
  "list_timestamp_alignment": {
    "GHI-onsite": {
      "timestamp_column": "GHI-onsite-timestamps",
      "length_mismatch_rows": 0,
      "timestamp_parse_failures": 0,
      "timestamp_elements": 614880,
      "offset_minutes_minimum": -4.0,
      "offset_minutes_median": -2.0,
      "offset_minutes_maximum": 0.0,
      "rows_with_a_timestamp_in_t_plus_5_to_t_plus_240": 0
    },
    "estimated_DNI-onsite": {
      "timestamp_column": "estimated_DNI-onsite-timestamps",
      "length_mismatch_rows": 0,
      "timestamp_parse_failures": 0,
      "timestamp_elements": 614880,
      "offset_minutes_minimum": -4.0,
      "offset_minutes_median": -2.0,
      "offset_minutes_maximum": 0.0,
      "rows_with_a_timestamp_in_t_plus_5_to_t_plus_240": 0
    },
    "estimated_DHI-onsite": {
      "timestamp_column": "estimated_DHI-onsite-timestamps",
      "length_mismatch_rows": 0,
      "timestamp_parse_failures": 0,
      "timestamp_elements": 614880,
      "offset_minutes_minimum": -4.0,
      "offset_minutes_median": -2.0,
      "offset_minutes_maximum": 0.0,
      "rows_with_a_timestamp_in_t_plus_5_to_t_plus_240": 0
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
      "GHI_mean-NWP_observe": 0.0,
      "GHI_mean-NWP_forecast": 0.27853134,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite": [
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
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:09.360",
        "1970-01-21T04:32:09.420",
        "1970-01-21T04:32:09.480",
        "1970-01-21T04:32:09.540",
        "1970-01-21T04:32:09.600"
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
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.280485903,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite": [
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
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:09.660",
        "1970-01-21T04:32:09.720",
        "1970-01-21T04:32:09.780",
        "1970-01-21T04:32:09.840",
        "1970-01-21T04:32:09.900"
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
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.282440467,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite": [
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
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:09.960",
        "1970-01-21T04:32:10.020",
        "1970-01-21T04:32:10.080",
        "1970-01-21T04:32:10.140",
        "1970-01-21T04:32:10.200"
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
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.28439503,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite": [
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
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:10.260",
        "1970-01-21T04:32:10.320",
        "1970-01-21T04:32:10.380",
        "1970-01-21T04:32:10.440",
        "1970-01-21T04:32:10.500"
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
      "GHI_mean-NWP_observe": null,
      "GHI_mean-NWP_forecast": 0.286349593,
      "estimated_DNI-onsite": [
        null,
        null,
        null,
        null,
        null
      ],
      "estimated_DHI-onsite": [
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
      "estimated_DNI-onsite-timestamps": [
        "1970-01-21T04:32:10.560",
        "1970-01-21T04:32:10.620",
        "1970-01-21T04:32:10.680",
        "1970-01-21T04:32:10.740",
        "1970-01-21T04:32:10.800"
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
  "usable": true
}
