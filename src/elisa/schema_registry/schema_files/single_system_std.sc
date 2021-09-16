{
  "type": "object",
  "required": [
    "system",
    "star"
  ],
  "properties": {
    "system": {
      "type": "object",
      "required": [
        "inclination",
        "rotation_period",
        "gamma"
      ],
      "properties": {
        "inclination": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "string"
            }
          ]
        },
        "rotation_period": {
          "anyOf": [
            {
              "type": "number",
              "minimum": 0.0
            },
            {
              "type": "string"
            }
          ]
        },
        "gamma": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "string"
            }
          ]
        },
        "reference_time": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "string"
            }
          ]
        },
        "phase_shift": {
          "type": "number"
        },
        "additional_light": {
          "type": "number"
        }
      }
    },
    "star": {
      "type": "object",
      "required": [
        "mass",
        "t_eff",
        "polar_log_g"
      ],
      "properties": {
        "mass": {"anyOf": [
          {"type": "number", "minimum": 0.0, "maximum": 150.0},
          {"type": "string"}
        ]},
        "t_eff": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "gravity_darkening": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "discretization_factor": {
          "type": "number",
          "minimum": 0.0
        },
        "albedo": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "metallicity": {
          "type": "number"
        },
        "atmosphere": {
          "type": "string"
        },
        "polar_log_g": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]}
      }
    }
  }
}
