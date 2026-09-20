"""生成意图分类训练数据 intent_train.csv（8 类）。"""
import csv, random
from pathlib import Path

random.seed(42)

TEMPLATES = {
    "policy_attendance": ["{x}是怎么规定的","{x}是什么","{x}怎么算","请问{x}","{x}在哪看","公司{x}"],
    "it_vpn": ["{x}怎么办","{x}连不上","{x}怎么弄","{x}报错","公司{x}","新电脑{x}"],
    "finance_expense": ["{x}标准是多少","{x}流程","{x}怎么报","{x}多久到账","请问{x}","{x}抬头"],
    "hr_onboarding": ["{x}要什么材料","{x}流程是什么","{x}多久办完","公司{x}","{x}缴纳比例","{x}证明"],
    "admin_logistics": ["{x}怎么预约","{x}在哪","{x}时间","请问{x}","公司{x}","{x}收费吗"],
    "engineering_eq": ["{x}怎么报修","{x}故障","{x}停机怎么办","{x}保养周期","请问{x}","{x}备件"],
    "legal_contract": ["{x}怎么签","{x}模板","{x}有效期","请问{x}","公司{x}","{x}审批流程"],
    "sales_marketing": ["{x}怎么申请","{x}标准","{x}流程","请问{x}","公司{x}","{x}折扣"],
}

PHRASES = {
    "policy_attendance": ["上下班时间","几点上班","打卡时间","年假天数","调休申请","加班审批","迟到怎么算","请假流程","补卡规则","午休时间"],
    "it_vpn": ["VPN连接","密码重置","电脑领用","打印机故障","内网访问","WiFi密码","软件安装","企业微信配置","电脑蓝屏","远程桌面"],
    "finance_expense": ["差旅报销","住宿费标准","发票抬头","电子发票要求","报销打款时间","工资条查看","出差补助","团建费用报销","个税计算","发票丢失"],
    "hr_onboarding": ["入职材料","试用期时长","社保缴纳","公积金比例","劳动合同期限","离职证明","年度体检","节日福利","转正申请","住房补贴"],
    "admin_logistics": ["食堂就餐","班车时刻","宿舍申请","快递收发","门禁权限","停车位","工牌补办","会议室预订","饮用水申请","公司活动报名"],
    "engineering_eq": ["产线故障报修","设备保养","PLC报警","模具维修","备件采购","停机处理","设备点检","产线换型","焊接不良","贴片机故障"],
    "legal_contract": ["合同模板","保密协议","专利申请","知识产权","公章使用","违约处理","法务咨询","劳动仲裁","竞业协议","商业秘密"],
    "sales_marketing": ["客户报价","渠道申请","品牌宣传","订单录入","回款跟进","展会报名","客户投诉","合同折扣","样品申请","市场调研"],
}

OUT = Path(__file__).resolve().parent.parent / "data" / "intent_train.csv"


def main():
    rows = [("label", "text")]
    for label, temps in TEMPLATES.items():
        for phrase in PHRASES[label]:
            for tpl in temps:
                rows.append((label, tpl.format(x=phrase)))
    random.shuffle(rows[1:])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    from collections import Counter
    c = Counter(r[0] for r in rows[1:])
    print(f"生成 {len(rows)-1} 条训练样本 -> {OUT}")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v} 条")


if __name__ == "__main__":
    main()
