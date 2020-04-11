{
  "type": "object",
  "required": [
    "system",
    "primary",
    "secondary"
  ],
  "properties": {
    "system": {
      "type": "object",
      "required": [
        "inclination",
        "period",
        "argument_of_periastron",
        "gamma",
        "eccentricity",
        "semi_major_axis",
        "mass_ratio"
      ],
      "properties": {
        "inclination": {
          "type": "number"
        },
        "period": {
          "type": "number"
        },
        "argument_of_periastron": {
          "type": "number"
        },
        "gamma": {
          "type": "number"
        },
        "eccentricity": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "semi_major_axis": {
          "type": "number",
          "minimum": 0.0
        },
        "mass_ratio": {
          "type": "number",
          "minimum": 0.0
        },
        "primary_minimum_time": {
          "type": "number"
        },
        "phase_shift": {
          "type": "number"
        },
        "additional_light": {
          "type": "number"
        }
      }
    },
    "primary": {
      "type": "object",
      "required": [
        "surface_potential",
        "synchronicity",
        "t_eff",
        "gravity_darkening",
        "albedo",
        "metallicity"
      ],
      "properties": {
        "surface_potential": {
          "type": "number",
          "minimum": 0.0
        },
        "synchronicity": {
          "type": "number",
          "minimum": 0.0
        },
        "t_eff": {
          "type": "number",
          "minimum": 3500.0,
          "maximum": 50000.0
        },
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
        "spots": {
          "type": "array",
          "items": {
            "type": "object",
            "required": [
              "longitude",
              "latitude",
              "angular_radius",
              "temperature_factor"
            ],
            "properties": {
              "longitude": {
                "type": "number",
                "minimum": 0,
                "maximum": 360
              },
              "latitude": {
                "type": "number",
                "minimum": 0,
                "maximum": 180
              },
              "angular_radius": {
                "type": "number"
              },
              "temperature_factor": {
                "type": "number",
                "minimum": 0
              },
              "discretization_factor": {
                "type": "number",
                "minimum": 0
              }
            }
          }
        },
        "pulsations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "l": {
                "type": "number"
              },
              "m": {
                "type": "number"
              },
              "amplitude": {
                "type": "number"
              },
              "frequency": {
                "type": "number"
              },
              "start_phase": {
                "type": "number"
              },
              "mode_axis_phi": {
                "type": "number"
              },
              "mode_axis_theta": {
                "type": "number"
              }
            }
          }
        }
      }
    },
    "secondary": {
      "type": "object",
      "required": [
        "surface_potential",
        "synchronicity",
        "t_eff",
        "gravity_darkening",
        "albedo",
        "metallicity"
      ],
      "properties": {
        "surface_potential": {
          "type": "number",
          "minimum": 0.0
        },
        "synchronicity": {
          "type": "number",
          "minimum": 0.0
        },
        "t_eff": {
          "type": "number",
          "minimum": 3500.0,
          "maximum": 50000.0
        },
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
        "spots": {
          "type": "array",
          "items": {
            "type": "object",
            "required": [
              "longitude",
              "latitude",
              "angular_radius",
              "temperature_factor"
            ],
            "properties": {
              "longitude": {
                "type": "number",
                "minimum": 0,
                "maximum": 360
              },
              "latitude": {
                "type": "number",
                "minimum": 0,
                "maximum": 180
              },
              "angular_radius": {
                "type": "number"
              },
              "temperature_factor": {
                "type": "number",
                "minimum": 0
              },
              "discretization_factor": {
                "type": "number",
                "minimum": 0
              }
            }
          }
        },
        "pulsations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "l": {
                "type": "number"
              },
              "m": {
                "type": "number"
              },
              "amplitude": {
                "type": "number"
              },
              "frequency": {
                "type": "number"
              },
              "start_phase": {
                "type": "number"
              },
              "mode_axis_phi": {
                "type": "number"
              },
              "mode_axis_theta": {
                "type": "number"
              }
            }
          }
        }
      }
    }
  }
}
