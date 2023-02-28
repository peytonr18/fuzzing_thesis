from textwrap import dedent

def copyright():
    print(dedent("""
            if (req->length < 4) {
        /* either this, or the server is printing bad messages,
           or the caller passed in garbage */
        ret = KRB5KRB_AP_ERR_MODIFIED;
        numresult = KRB5_KPASSWD_MALFORMED;
        strlcpy(strresult, "Request was truncated", sizeof(strresult));
        goto chpwfail;
    }

    ptr = req->data;
    /* verify length */
    plen = (*ptr++ & 0xff);
    plen = (plen<<8) | (*ptr++ & 0xff);
    if (plen != req->length) {
        ret = KRB5KRB_AP_ERR_MODIFIED;
        numresult = KRB5_KPASSWD_MALFORMED;
        strlcpy(strresult, "Request length was inconsistent",
                sizeof(strresult));
        goto chpwfail;
    }

    /* verify version number */
    vno = (*ptr++ & 0xff) ;
    vno = (vno<<8) | (*ptr++ & 0xff);
    if (vno != 1 && vno != RFC3244_VERSION) {
        ret = KRB5KDC_ERR_BAD_PVNO;
        numresult = KRB5_KPASSWD_BAD_VERSION;
        snprintf(strresult, sizeof(strresult),
                 "Request contained unknown protocol version number %d", vno);
        goto chpwfail;
    }

    /* read, check ap-req length */
    ap_req.length = (*ptr++ & 0xff);
    ap_req.length = (ap_req.length<<8) | (*ptr++ & 0xff);
    if (ptr + ap_req.length >= req->data + req->length) {
        ret = KRB5KRB_AP_ERR_MODIFIED;
        numresult = KRB5_KPASSWD_MALFORMED;
        strlcpy(strresult, "Request was truncated in AP-REQ",
                sizeof(strresult));
        goto chpwfail;
    }
    """).strip("\n"))
