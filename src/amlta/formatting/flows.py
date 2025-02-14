from amlta.probas.flows import Flow


def flow_repr(flow: Flow) -> str:
    return 'Flow("{name}" {direction}, {amount:g} {unit} ({property_name}), type={type_of_flow!r}, class={classification_hierarchy!r})'.format(
        name=flow.flow_description,
        direction=flow.exchange_direction.lower()
        if flow.exchange_direction
        else "unknown",
        type_of_flow=flow.exchange_type_of_flow,
        amount=flow.exchange_resulting_amount,
        classification_hierarchy=flow.exchange_classification_hierarchy,
        property_name=flow.flow_property_name,
        unit=flow.flow_property_unit,
    )
