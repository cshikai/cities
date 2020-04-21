# *- coding: utf-8 -*-
import hashlib

from django.utils.encoding import smart_str
from requests import request


class DianpingApiError(Exception):
    def __unicode__(self):
        return self.message

    def __str__(self):
        return smart_str(self.message)


class DianpingApi(object):
    API_URL_BASE = 'http://api.dianping.com/v1/'
    REQUESTS_TIMEOUT = 30
    USER_AGENT = 'python-dianping-api 0.0.1'

    def __init__(self, appkey, secret):
        self.appkey = appkey
        self.secret = secret

    def request(self, api_url, data={}, **kwargs):
        kwargs.setdefault('headers', {})
        kwargs.setdefault('timeout', self.REQUESTS_TIMEOUT)

        if 'User-Agent' not in kwargs['headers']:
            kwargs['headers']['User-Agent'] = self.USER_AGENT

        h = hashlib.sha1()
        h.update(self.appkey)
        for k in sorted(data.iterkeys()):
            h.update(k)
            h.update(smart_str(data[k]))
        h.update(self.secret)

        payload = {'appkey': self.appkey, 'sign': h.hexdigest().upper()}
        payload.update(data)
        print payload

        url = self.API_URL_BASE + api_url
        response = request('GET', url, params=payload, **kwargs)
        response.raise_for_status()
        result = response.json()
        status = result.pop('status', 'ERROR').upper()
        if status == 'ERROR':
            error = result.get(
                'error', {'errorCode': -1, 'errorMessage': 'unknown'})
            raise DianpingApiError(
                '%s - %s' % (error['errorCode'], error['errorMessage']))
        return result

    def get_cities(self):
        result = self.request('metadata/get_cities_with_businesses')
        return result['cities']

    def get_regions(self, city=None):
        data = {}
        if city:
            data['city'] = city
        result = self.request('metadata/get_regions_with_businesses', data)
        return result['cities']

    def get_categories(self, city=None):
        data = {}
        if city:
            data['city'] = city
        result = self.request('metadata/get_categories_with_businesses', data)
        return result['categories']

    def find_businesses(self, **kwargs):
        '''sort:
        1: 默认
        2: 星级高优先
        3: 产品评价高优先
        4: 环境评价高优先
        5: 服务评价高优先
        6: 点评数量多优先
        7: 离传入经纬度坐标距离近优先
        8: 人均价格低优先
        9: 人均价格高优先
        '''
        kwargs['limit'] = 40
        category = kwargs.pop('category', [])
        data = {}
        data.update(kwargs)
        if category:
            if isinstance(category, list):
                categories = ','.join(category[:5])
            else:
                categories = category
            data['category'] = categories
        result = self.request('business/find_businesses', data)
        return result['businesses']

    def get_batch_businesses_by_id(self, business_ids):
        if isinstance(business_ids, list):
            _ids = ','.join(business_ids[:40])
        else:
            _ids = business_ids
        data = {'business_ids': _ids}
        result = self.request('business/get_batch_businesses_by_id', data)
        return result['businesses']

    def get_recent_reviews(self, business_id):
        result = self.request(
            'review/get_recent_reviews', {'business_id': business_id})
        return result['reviews']

    def get_single_business(self, business_id):
        data = {'business_id': business_id}
        result = self.request('business/get_single_business', data)
        return result
