class Bond:
    def get_price(self, coupon, face_value, int_rate, years, freq=2):
        total_coupon_pv = self.get_coupons_pv(coupon, int_rate, years, freq)
        face_value_pv = self.get_face_value_pv(face_value, int_rate, years)
        return total_coupon_pv + face_value_pv

    @staticmethod
    def get_face_value_pv(face_value, int_rate, years):
        return face_value / (1 + int_rate) ** years

    def get_coupons_pv(self, coupon, int_rate, years, freq=2):
        pv = 0
        for period in range(years * freq):
            pv += self.get_coupon_pv(coupon, int_rate, period + 1, freq)
        return pv

    @staticmethod
    def get_coupon_pv(coupon, int_rate, period, freq=2):
        return (coupon / freq) / (1 + int_rate / freq) ** period

    def get_ytm(self, bond_price, face_value, coupon, years, freq=2, estimate=0.05):
        import scipy
        from scipy import optimize

        get_yield = (
            lambda int_rate: self.get_price(coupon, face_value, int_rate, years, freq)
            - bond_price
        )
        return optimize.newton(get_yield, estimate)
