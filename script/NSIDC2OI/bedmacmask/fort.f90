subroutine remap_mask_nsidc2oi(mask_ns, mask_oi, lat_ns, lon_ns, lat_oi, lon_oi)
    implicit none


    integer(kind=1), dimension(13333, 13333), intent(in) :: mask_ns
    integer(kind=1), dimension(1440, 720), intent(out) :: mask_oi
    real   (kind=4), dimension(13333, 13333), intent(in ) :: lat_ns, lon_ns
    real   (kind=4), dimension( 720), intent(in ) :: lat_oi
    real   (kind=4), dimension(1440), intent(in ) :: lon_oi

!! f2py instructions for compile use which is needed, for f90 free-stype not to use Cf2py or it'll fail.
!! see how it's been called by Python.
!f2py intent(in) mask_ns
!f2py intent(out) mask_oi
!f2py intent(in) lat_ns, lon_ns, lat_oi, lon_oi

    ! local vars
    integer :: i_ns, j_ns, i_oi, j_oi, flag, n_land, n_ocn, n_ice
    real    :: lat, lon 
    integer(kind=4), dimension(1440, 720, 5) :: cnt

    mask_oi = 100
    cnt = 0

    do j_ns = 1, 13333; do i_ns = 1, 13333
        lat = lat_ns(i_ns, j_ns) 
        lon = lon_ns(i_ns, j_ns)
        if (lon < 0) then
            lon = lon + 360
        end if
        
        j_oi = nint(4*(lat+89.875) + 1)
        if (j_oi < 1) then
            j_oi = 1
        end if

        i_oi = nint(4*(lon-0.125) + 1)
        if (i_oi < 1) then
            i_oi = 1440
        else if (i_oi > 1440) then
            i_oi = 1
        end if

        flag = mask_ns(i_ns, j_ns)
        cnt(i_oi, j_oi, flag+1) = cnt(i_oi, j_oi, flag+1) + 1
    end do; end do

    do j_oi = 1, 720; do i_oi = 1, 1440
        n_land = cnt(i_oi, j_oi, 2) + cnt(i_oi, j_oi, 3) + cnt(i_oi, j_oi, 5)
        n_ocn  = cnt(i_oi, j_oi, 1)
        n_ice  = cnt(i_oi, j_oi, 4)

        if (n_ice > 0 .and. n_ice >= n_land .and. n_ice >= n_ocn) then
            mask_oi(i_oi, j_oi) = 3
        else if (n_land > 0 .and. n_land >= n_ocn .and. n_land >= n_ice) then
            mask_oi(i_oi, j_oi) = 2
        else if (n_ocn > 0 .and. n_ocn >= n_ice .and. n_ocn >= n_land) then
            mask_oi(i_oi, j_oi) = 0
        end if

        if (j_oi <= 30 .and. mask_oi(i_oi, j_oi) == 100) then
            mask_oi(i_oi, j_oi) = 2
        end if
    end do; end do

    return
end subroutine remap_mask_nsidc2oi
