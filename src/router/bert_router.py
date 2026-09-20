"""第 2 层路由：BERT 意图分类器（复用你已有的 bert-base-chinese 微调代码）。
未训练前，此模块返回 None，自动降级到 LLM 路由。
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .rule_router import RouteResult


class BertRouter:
    def __init__(self,
                 model_name: str = "bert-base-chinese",
                 ckpt_path: str = "",
                 label_list: Optional[List[str]] = None,
                 kb_map: Optional[Dict[str, str]] = None,
                 device: str = "cuda",
                 confidence_threshold: float = 0.85):
        self.conf_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        # 默认与 data/class.txt、checkpoints/intent_bert/best.pt（9 类）保持一致
        self.label_list = label_list or [
            "policy_attendance", "it_vpn", "finance_expense", "hr_onboarding",
            "admin_logistics", "engineering_eq", "legal_contract", "sales_marketing",
            "other_chitchat",
        ]
        self.kb_map = kb_map or {
            "policy_attendance": "policy", "it_vpn": "it",
            "finance_expense": "finance", "hr_onboarding": "hr",
            "admin_logistics": "admin", "engineering_eq": "engineering",
            "legal_contract": "legal", "sales_marketing": "sales",
        }
        # 初始化模型结构；若有 checkpoint 再加载权重
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.label_list)
        )
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def route(self, query: str) -> Optional[RouteResult]:
        """返回分类结果；置信度低于阈值时返回 None，交给上层 LLM 兜底。"""
        enc = self.tokenizer(query, return_tensors="pt",
                             truncation=True, max_length=64, padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        if conf < self.conf_threshold:
            return None
        intent = self.label_list[idx]
        return RouteResult(intent=intent, kb_dir=self.kb_map.get(intent, ""),
                           source="bert", confidence=conf)


if __name__ == "__main__":
    # 未训练时直接运行会加载 bert-base-chinese 随机分类头，仅用于验证接口
    br = BertRouter(device="cpu")
    print(br.route("请问公司年假怎么算"))
