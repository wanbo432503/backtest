#!/usr/bin/env python3
"""
多市场股票搜索API测试脚本
用于验证新添加的API端点是否正常工作
"""

import requests
import json
from typing import Dict, List

# API基础URL（根据实际部署调整）
BASE_URL = "http://localhost:8005"

def check_api_endpoint(endpoint: str, params: dict, description: str = "") -> bool:
    """测试API端点"""
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"\n{'='*60}")
        print(f"🔍 测试: {description}")
        print(f"   URL: {url}")
        print(f"   参数: {params}")
        print('='*60)
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功 (200)")
            print(f"   找到 {data.get('count', 0)} 个结果")
            
            # 打印前2个结果
            results = data.get('results', [])
            for i, result in enumerate(results[:2], 1):
                market = result.get('market', 'N/A')
                symbol = result.get('symbol', 'N/A')
                name = result.get('name', 'N/A')
                print(f"   {i}. [{market}] {symbol:15s} - {name[:40]}")
            
            return True
        else:
            print(f"❌ 失败 ({response.status_code})")
            print(f"   错误: {response.json().get('detail', '未知错误')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 连接失败 - 请确保FastAPI服务正在运行 (http://localhost:8005)")
        return False
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False

def run_tests():
    """运行所有测试"""
    print("\n")
    print("🚀 " + "="*58 + " 🚀")
    print("   多市场股票搜索API测试套件")
    print("🚀 " + "="*58 + " 🚀")
    
    test_cases = [
        # 美股搜索
        {
            "endpoint": "/search-us-stocks",
            "params": {"query": "苹果"},
            "description": "美股搜索: 苹果"
        },
        {
            "endpoint": "/search-us-stocks",
            "params": {"query": "MSFT"},
            "description": "美股搜索: MSFT (美股代码)"
        },
        
        # A股搜索
        {
            "endpoint": "/search-cn-stocks",
            "params": {"query": "茅台"},
            "description": "A股搜索: 茅台"
        },
        {
            "endpoint": "/search-cn-stocks",
            "params": {"query": "平安"},
            "description": "A股搜索: 平安"
        },
        {
            "endpoint": "/search-cn-stocks",
            "params": {"query": "SH600519"},
            "description": "A股搜索: SH600519 (A股代码)"
        },
        
        # 港股搜索
        {
            "endpoint": "/search-hk-stocks",
            "params": {"query": "腾讯"},
            "description": "港股搜索: 腾讯"
        },
        {
            "endpoint": "/search-hk-stocks",
            "params": {"query": "0700.HK"},
            "description": "港股搜索: 0700.HK (港股代码)"
        },
        {
            "endpoint": "/search-hk-stocks",
            "params": {"query": "小米"},
            "description": "港股搜索: 小米"
        },
        
        # 多市场搜索
        {
            "endpoint": "/search-stocks-multi-market",
            "params": {"query": "阿里巴巴"},
            "description": "多市场搜索: 阿里巴巴 (所有市场)"
        },
        {
            "endpoint": "/search-stocks-multi-market",
            "params": {"query": "百度", "market": "US"},
            "description": "多市场搜索: 百度 (仅美股)"
        },
        {
            "endpoint": "/search-stocks-multi-market",
            "params": {"query": "腾讯", "market": "HK"},
            "description": "多市场搜索: 腾讯 (仅港股)"
        },
        {
            "endpoint": "/search-stocks-multi-market",
            "params": {"query": "工商银行", "market": "CN"},
            "description": "多市场搜索: 工商银行 (仅A股)"
        },
        
        # 原始API（兼容性测试）
        {
            "endpoint": "/search-stocks",
            "params": {"query": "微软"},
            "description": "原始API: 微软 (向后兼容)"
        }
    ]
    
    results = []
    for test_case in test_cases:
        success = check_api_endpoint(
            test_case["endpoint"],
            test_case["params"],
            test_case["description"]
        )
        results.append({
            "test": test_case["description"],
            "success": success
        })
    
    # 打印总结
    print(f"\n\n{'='*60}")
    print("📊 测试总结")
    print('='*60)
    
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    
    print(f"\n总测试数: {total}")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"成功率: {(passed/total*100):.1f}%")
    
    # 详细列表
    print(f"\n{'='*60}")
    print("详细结果:")
    print('='*60)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status:10s} | {result['test']}")
    
    # 最终状态
    print(f"\n{'='*60}")
    if failed == 0:
        print("🎉 所有测试通过！API工作正常。")
    else:
        print(f"⚠️  有 {failed} 个测试失败，请检查API服务状态。")
    print('='*60 + "\n")
    
    return failed == 0

if __name__ == "__main__":
    import sys
    
    print("\n提示: 请确保FastAPI服务正在运行")
    print("启动命令: python main.py")
    print("服务地址: http://localhost:8005")
    print("API文档: http://localhost:8005/docs")
    
    success = run_tests()
    sys.exit(0 if success else 1)
