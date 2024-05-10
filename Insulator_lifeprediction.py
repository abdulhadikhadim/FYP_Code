class Insulator:
    def __init__(self, leakage_current, applied_voltage, temperature, humidity):
        self._leakage_current = leakage_current
        self._applied_voltage = applied_voltage
        self._temperature = temperature
        self._humidity = humidity
    @property
    def leakage_current(self):
        return self._leakage_current

    @property
    def applied_voltage(self):
        return self._applied_voltage

    @property
    def temperature(self):
        return self._temperature

    @property
    def humidity(self):
        return self._humidity

    def calculate_life_left(self, max_leakage_current=10, max_applied_voltage=1000,
                            life_reduction_factor_per_degree=0.02,
                            life_reduction_factor_per_percent_humidity=0.01, reference_temperature=20):
        leakage_ratio = min(1, self.leakage_current / max_leakage_current)
        voltage_ratio = min(1, self.applied_voltage / max_applied_voltage)

        life_left_percentage = (leakage_ratio + voltage_ratio) * 50  # Equal weightage to leakage and voltage
        life_left_percentage -= (self.temperature - reference_temperature) * life_reduction_factor_per_degree
        life_left_percentage -= self.humidity * life_reduction_factor_per_percent_humidity

        return max(0, life_left_percentage)  # Ensure life left is not negative


def get_valid_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid floating-point number.")


# Example usage:
leakage = get_valid_float_input("Enter leakage current: ")
voltage = get_valid_float_input("Enter applied voltage: ")
temperature = get_valid_float_input("Enter temperature (in Celsius): ")
humidity = get_valid_float_input("Enter humidity (in percentage): ")

insulator = Insulator(leakage_current=leakage, applied_voltage=voltage, temperature=temperature, humidity=humidity)
life_left = insulator.calculate_life_left()
print(f"Life left: {life_left}%")
