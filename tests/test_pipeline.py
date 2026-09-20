"""pytest 冒烟测试：不依赖真实模型，只测规则路由。
运行：pytest tests/ -v
"""
from src.router import RuleRouter


def test_rule_router_attendance():
    r = RuleRouter("config/intents.yaml")
    res = r.route("公司上下班时间是几点")
    assert res is not None
    assert res.intent == "policy_attendance"
    assert res.kb_dir == "policy"
    assert res.source == "rule"


def test_rule_router_it():
    r = RuleRouter("config/intents.yaml")
    res = r.route("VPN 连不上")
    assert res is not None
    assert res.intent == "it_vpn"


def test_rule_router_no_hit():
    r = RuleRouter("config/intents.yaml")
    assert r.route("量子纠缠是什么") is None
