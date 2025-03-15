from app.agent.base import BaseAgent
from app.flow.base import BaseFlow, FlowType
from app.flow.planning import PlanningFlow


class FlowFactory:
    """複数のエージェントをサポートする異なるタイプのフローを作成するためのファクトリー"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: BaseAgent | list[BaseAgent] | dict[str, BaseAgent],
        **kwargs,
    ) -> BaseFlow:
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }

        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"不明なフロータイプです: {flow_type}")

        return flow_class(agents, **kwargs)
