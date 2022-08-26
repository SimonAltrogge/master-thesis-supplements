import collections
import warnings
import numpy as np
from neuralnetwork.neuronpopulation import SpikeType


class History:
    def __init__(self, variables_to_record=None, maxlen=None):
        self.maxlen = maxlen

        if variables_to_record is None:
            self.variables_to_record = set()
        elif isinstance(variables_to_record, str):
            self.variables_to_record = set(variables_to_record.split())
        else:
            self.variables_to_record = set(variables_to_record)

        if len(self.variables_to_record) == 0:
            raise ValueError("History is not given any variables to record")

        for variable in self.variables_to_record:
            setattr(self, variable, collections.deque(maxlen=maxlen))

    def __len__(self):
        first_variable, *_ = self.variables_to_record

        return len(self.values(first_variable))

    def values(self, variable):
        return getattr(self, variable)

    def append(self, **variables_values):
        for variable in self.variables_to_record:
            if variable in variables_values:
                value = variables_values[variable]
            else:
                value = None
                warnings.warn(
                    f"History should record {variable} but no {variable} value was given"
                )

            self.values(variable).append(value)

        if not variables_values.keys() <= self.variables_to_record:
            raise TypeError("variable given to History that should not be recorded")


class SpikeHistory(History):
    def __init__(self, neuron_population, start_time=0.0):
        super().__init__({"time", "neuron", "type"})

        self._neuron_population = neuron_population
        self._time = start_time

        self.count = 0
        self.count_by_type = {spike_type: 0 for spike_type in SpikeType}
        self.count_by_neuron = np.zeros(self._neuron_population.neuron_count, dtype=int)
        self.count_by_type_by_neuron = np.array(
            [
                {spike_type: 0 for spike_type in SpikeType}
                for _ in range(self._neuron_population.neuron_count)
            ]
        )

        self._monitor(self._neuron_population)

    def __len__(self):
        return self.count

    def append(self, **variables_values):
        super().append(**variables_values)

        self.count += 1

        has_type = "type" in variables_values
        if has_type:
            self.count_by_type[variables_values["type"]] += 1

        has_neuron = "neuron" in variables_values
        if has_neuron:
            self.count_by_neuron[variables_values["neuron"]] += 1

        if has_type and has_neuron:
            self.count_by_type_by_neuron[variables_values["neuron"]][
                variables_values["type"]
            ] += 1

    @property
    def spike_trains(self):
        return [
            np.asarray(self.values("time"))[np.equal(self.values("neuron"), neuron)]
            for neuron in range(self._neuron_population.neuron_count)
        ]

    def _monitor(self, neuron_population):
        original_draw_next_spike = neuron_population.draw_next_spike

        def monitored_draw_next_spike(return_spike_type=False):
            interspike_interval, spiking_neuron, spike_type = original_draw_next_spike(
                return_spike_type=True
            )

            self._time += interspike_interval
            self.append(
                time=self._time,
                neuron=spiking_neuron,
                type=spike_type,
            )

            if return_spike_type:
                return (interspike_interval, spiking_neuron, spike_type)

            return (interspike_interval, spiking_neuron)

        neuron_population.draw_next_spike = monitored_draw_next_spike
