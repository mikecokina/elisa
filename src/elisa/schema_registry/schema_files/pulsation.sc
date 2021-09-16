{
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "l",
      "m",
      "amplitude",
      "frequency"
    ],
    "properties": {
      "l": {"type": "integer"},
      "m": {"type": "integer"},
      "amplitude": {
        "anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]
      },
      "frequency": {
        "anyOf": [
          {
            "type": "number",
            "minimum": 0},
          {"type": "string"}
        ]
      },
      "start_phase": {
        "anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]
      },
      "mode_axis_phi": {
        "anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]
      },
      "mode_axis_theta": {
        "anyOf": [
          {
            "type": "number",
            "minimum": 0,
            "maximum": 180},
          {"type": "string"}
        ]
      },
      "temperature_perturbation_phase_shift": {
        "anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]
      },
      "horizontal_to_radial_amplitude_ratio": {
        "type": "number",
        "minimum": 0
      },
      "temperature_amplitude_factor": {
        "type": "number"
      },
      "tidally_locked": {
        "type": "boolean"
      }
    }
  }
}